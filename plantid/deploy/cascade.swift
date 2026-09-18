// The cascade, in Swift. A port of what `tests/test_export_ios.py` asserts is a
// complete specification of `narrowcast.predict.Bundle` — that test compares this
// arithmetic, written in Python, against the tool itself over every row of a real
// Oregon bundle and requires zero disagreements.
//
// UNBUILT. This was written alongside the export and its Python twin is tested;
// it has never been compiled. Treat compiler errors as typos, and treat any
// *behavioural* difference from the Python twin as a bug here, since that twin is
// the one with the passing parity test.
//
// What the encoder does and what you must not repeat
// --------------------------------------------------
// The .mlpackage carries the 1/255 scale, the channel normalisation and the final
// L2 inside its graph. Outside it, all you do is resize and centre-crop to
// `coreml.side` with `preprocess.interpolation`, and hand over RGB. Normalising
// again, or skipping the crop, puts the vector in a different space from the one
// the head was fitted on — the failure mode that shows cosine ~1.0 against itself
// while agreeing with nothing.

import Foundation

// Named `CascadeBundle`, not `Bundle`: Foundation already has a `Bundle` and
// shadowing it breaks `Bundle.main` in the same file.
struct CascadeBundle: Decodable {
    struct B64: Decodable { let b64: String; let shape: [Int] }
    struct CoreMLSpec: Decodable { let package: String; let input: String
                                   let side: Int; let output: String; let dim: Int }
    struct Pre: Decodable { let side: Int; let interpolation: String }
    struct Thresholds: Decodable { let t_group: Double; let t_label: Double
                                   let t_novel: Double? }

    let schema: Int
    let encoder: String
    let coreml: CoreMLSpec
    let preprocess: Pre
    let classes: [String]
    let reject_class: String?
    let coef: B64
    let intercept: B64
    let groups: [String: String]
    let thresholds: Thresholds
    let never_answer: [String]
}

func floats(_ a: CascadeBundle.B64) -> [Float] {
    guard let d = Data(base64Encoded: a.b64) else { return [] }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }  // little-endian
}

enum Rank: String { case label, group, decline }
struct Answer { let rank: Rank; let text: String?; let labelConf: Double
                let groupConf: Double; let novelty: Double }

final class Cascade {
    private let b: CascadeBundle
    private let coef: [Float], intercept: [Float]
    private let keep: [Int]          // indices of answerable classes (not __OTHER__)
    private let groupOf: [String]    // per kept class
    private let uniqueGroups: [String]

    init(_ b: CascadeBundle) {
        self.b = b
        coef = floats(b.coef); intercept = floats(b.intercept)
        keep = b.classes.indices.filter { b.classes[$0] != b.reject_class }
        // The bundle's own map. Never `label.split(" ")[0]` — that is a
        // Latin-binomial convention and has been wrong in five places; a foraging
        // list groups by family, where it is wrong every time.
        groupOf = keep.map { b.groups[b.classes[$0]] ?? b.classes[$0] }
        uniqueGroups = Array(Set(groupOf)).sorted()
    }

    /// `embedding` is the .mlpackage's output: already unit-norm.
    func callAsFunction(_ embedding: [Float]) -> Answer {
        let d = b.coreml.dim, n = b.classes.count

        // Softmax over EVERY class, including the reject class. Masking first and
        // then softmaxing renormalises the reject mass away and silently deletes
        // the model's ability to say "none of these".
        var logits = [Double](repeating: 0, count: n)
        for c in 0..<n {
            var s = Double(intercept[c])
            for j in 0..<d { s += Double(coef[c * d + j]) * Double(embedding[j]) }
            logits[c] = s
        }
        let mx = logits.max() ?? 0
        let exps = logits.map { exp($0 - mx) }
        let total = exps.reduce(0, +)
        let proba = exps.map { $0 / max(total, 1e-12) }

        // ...and only now drop it from the answerable set.
        let cata = keep.map { proba[$0] }
        let inList = cata.reduce(0, +)
        let novelty = inList / max(proba.reduce(0, +), 1e-12)   // 1 - P(__OTHER__)

        var groupMass = [String: Double]()
        for (i, g) in groupOf.enumerated() { groupMass[g, default: 0] += cata[i] }

        let bestLabel = cata.indices.max { cata[$0] < cata[$1] } ?? 0
        let labelConf = cata[bestLabel]
        let predLabel = b.classes[keep[bestLabel]]
        let best = groupMass.max { $0.value < $1.value }
        let predGroup = best?.key ?? ""
        let groupConf = best?.value ?? 0

        // Order is load-bearing and matches `cascade.decide`: retreat, then
        // decline, then the near-OOD gate — which declines unconditionally.
        var rank = Rank.label
        if labelConf < b.thresholds.t_label { rank = .group }
        if groupConf < b.thresholds.t_group { rank = .decline }
        if let tn = b.thresholds.t_novel, novelty < tn { rank = .decline }

        // Suppression last. It removes the look-alike, not the hazard.
        let never = Set(b.never_answer)
        if !never.isEmpty {
            if rank == .label, never.contains(predLabel) {
                rank = .decline
            } else if rank == .group {
                let members = Set(groupOf.indices.filter { groupOf[$0] == predGroup }
                                          .map { b.classes[keep[$0]] })
                if !members.isEmpty, members.isSubset(of: never) { rank = .decline }
            }
        }

        let text: String? = rank == .label ? predLabel : (rank == .group ? predGroup : nil)
        return Answer(rank: rank, text: text, labelConf: labelConf,
                      groupConf: groupConf, novelty: novelty)
    }
}

extension Cascade {
    /// Load from `bundle.json` in the app bundle. One obvious entry point.
    static func load(resource: String = "bundle") throws -> Cascade {
        guard let url = Foundation.Bundle.main.url(forResource: resource,
                                                   withExtension: "json") else {
            throw NSError(domain: "Cascade", code: 1, userInfo:
                [NSLocalizedDescriptionKey: "\(resource).json not in the app bundle"])
        }
        let b = try JSONDecoder().decode(CascadeBundle.self,
                                         from: try Data(contentsOf: url))
        return Cascade(b)
    }
}
