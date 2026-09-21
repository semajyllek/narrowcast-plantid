import Accelerate
import Foundation

// "My list" mode: the user names their species and the model is fitted here, on
// the phone, from a shipped bank of embeddings.
//
// This is not an approximation of the offline pipeline. `build.fit_head` never
// sees an image either — it takes embeddings — so fitting from stored ones is
// the same computation. Measured: nearest-centroid is within a point of logistic
// regression at every exemplar count (0.967 vs 0.973 at eight), which is why
// this is a mean and a dot product rather than a port of L-BFGS.
//
// Why bother, when the app also ships a 901-species head? Because on the species
// a user actually chose, narrow wins: 0.963 against 0.919, measured over ten
// random selections of twenty. Wide is better at "what is this"; narrow is
// better at "is this one of mine", and for foraging narrow is the only
// defensible one.

struct BankIndex: Decodable {
    let dim: Int
    let count: Int
    let temperature: Float
    let species: [String]
    let common: [String]
    let offset: [Int]
    let n_exemplars: [Int]
    let n_obs: [Int]
}

struct Bank {
    let index: BankIndex
    let rows: [Float]                 // n_rows * dim, unit-norm, widened from f16

    init(indexURL: URL, blobURL: URL) throws {
        index = try JSONDecoder().decode(BankIndex.self, from: Data(contentsOf: indexURL))
        let raw = try Data(contentsOf: blobURL)
        rows = raw.withUnsafeBytes { buf in
            let p = buf.bindMemory(to: Float16.self)
            return (0..<p.count).map { Float(p[$0]) }
        }
    }

    func exemplars(of species: Int) -> [ArraySlice<Float>] {
        let d = index.dim, o = index.offset[species]
        return (0..<index.n_exemplars[species]).map {
            rows[(o + $0) * d ..< (o + $0 + 1) * d]
        }
    }

    /// Unit-norm mean of a species' plant-exemplars, optionally leaving one out.
    func centroid(of species: Int, excluding drop: Int? = nil) -> [Float] {
        let d = index.dim
        var m = [Float](repeating: 0, count: d)
        var n = 0
        for (k, e) in exemplars(of: species).enumerated() where k != drop {
            for (j, v) in e.enumerated() { m[j] += v }
            n += 1
        }
        guard n > 0 else { return m }
        var norm: Float = 0
        for j in 0..<d { m[j] /= Float(n); norm += m[j] * m[j] }
        norm = max(norm.squareRoot(), 1e-12)
        for j in 0..<d { m[j] /= norm }
        return m
    }
}

/// A model over the user's chosen species, fitted from the bank.
struct LocalModel {
    let bank: Bank
    let selection: [Int]              // indices into bank.index.species
    let centroids: [[Float]]          // ALL species — the unchosen ones are the
                                      // reject class, and they are better
                                      // negatives than a background pool because
                                      // they are the actual deployment set
    let genusOf: [String]
    let tGroup: Float
    let tLabel: Float
    /// `centroids` laid out contiguously, so BLAS can see it.
    let flatCentroids: [Float]

    init(bank: Bank, selection: [Int], centroids: [[Float]], genusOf: [String],
         tGroup: Float, tLabel: Float) {
        self.bank = bank; self.selection = selection; self.centroids = centroids
        self.genusOf = genusOf; self.tGroup = tGroup; self.tLabel = tLabel
        self.flatCentroids = centroids.flatMap { $0 }
    }

    var names: [String] { selection.map { bank.index.species[$0] } }

    struct Scores { let label: Float; let group: Float; let novelty: Float
                    let best: Int; let bestGenus: String }

    /// Cosine to every species, softmax, then read the chosen subset.
    ///
    /// The similarities go through BLAS. In plain Swift loops, fitting scored
    /// 6,578 calibration rows against 961 centroids — 4.8 billion multiply-adds —
    /// and took 1.9 seconds on a laptop, which is not a thing to do while someone
    /// waits. `cblas_sgemv` is the same arithmetic on the vector unit.
    func score(_ embedding: [Float]) -> Scores {
        let T = bank.index.temperature
        let d = bank.index.dim, n = centroids.count
        var sims = [Float](repeating: 0, count: n)
        flatCentroids.withUnsafeBufferPointer { M in
            embedding.withUnsafeBufferPointer { x in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(n), Int32(d), 1.0,
                            M.baseAddress, Int32(d), x.baseAddress, 1, 0.0,
                            &sims, 1)
            }
        }
        let mx = sims.max() ?? 0
        var exps = sims.map { expf(T * ($0 - mx)) }
        let total = max(exps.reduce(0, +), 1e-12)
        for i in exps.indices { exps[i] /= total }

        var inList: Float = 0, bestP: Float = -1, best = 0
        var byGenus = [String: Float]()
        for (k, i) in selection.enumerated() {
            let p = exps[i]
            inList += p
            byGenus[genusOf[i], default: 0] += p
            if p > bestP { bestP = p; best = k }
        }
        let g = byGenus.max { $0.value < $1.value }
        return Scores(label: bestP, group: g?.value ?? 0, novelty: inList,
                      best: best, bestGenus: g?.key ?? "")
    }

    func answer(_ embedding: [Float]) -> (rank: Rank, text: String?) {
        let s = score(embedding)
        var rank = Rank.label
        if s.label < tLabel { rank = .group }
        if s.group < tGroup { rank = .decline }
        return (rank, rank == .label ? names[s.best]
                    : rank == .group ? s.bestGenus : nil)
    }

    /// Fit the two thresholds from the bank itself, leave-one-plant-out.
    ///
    /// The positives are each chosen species' exemplars scored against a centroid
    /// built *without* them — so they are honest held-out rows, not the rows the
    /// centroid was made from. The negatives are the species the user did not
    /// choose. Both come free with the bank, which is the whole point of shipping
    /// it: the operating point is fitted to *this* selection rather than assumed
    /// from a table, and twenty congeners need a different one from twenty
    /// unrelated plants.
    static func fit(bank: Bank, selection: [Int], utility: Utility = .identify,
                    pOOD: Float = 0.5) -> LocalModel {
        let genusOf = bank.index.species.map { String($0.split(separator: " ")[0]) }
        var centroids = (0..<bank.index.count).map { bank.centroid(of: $0) }

        // calibration rows: (labelConf, groupConf, isCorrect, isInList)
        var cal: [(Float, Float, Bool, Bool)] = []
        let sel = Set(selection)
        let probe = LocalModel(bank: bank, selection: selection, centroids: centroids,
                               genusOf: genusOf, tGroup: 0, tLabel: 0)
        for (k, i) in selection.enumerated() where bank.index.n_exemplars[i] >= 2 {
            for drop in 0..<bank.index.n_exemplars[i] {
                let held = Array(bank.exemplars(of: i)[drop])
                let saved = centroids[i]
                centroids[i] = bank.centroid(of: i, excluding: drop)
                let m = LocalModel(bank: bank, selection: selection, centroids: centroids,
                                   genusOf: genusOf, tGroup: 0, tLabel: 0)
                let s = m.score(held)
                cal.append((s.label, s.group, s.best == k, true))
                centroids[i] = saved
            }
        }
        for i in 0..<bank.index.count where !sel.contains(i) {
            for e in bank.exemplars(of: i) {
                let s = probe.score(Array(e))
                cal.append((s.label, s.group, false, false))
            }
        }

        let (tg, tl) = gridSearch(cal, utility: utility, pOOD: pOOD)
        return LocalModel(bank: bank, selection: selection, centroids: centroids,
                          genusOf: genusOf, tGroup: tg, tLabel: tl)
    }
}

/// Declared payoffs. Fixed in source, chosen before fitting, never tuned to an
/// outcome — the discipline the whole cascade rests on.
struct Utility {
    let labelCorrect: Float, groupCorrect: Float, wrong: Float
    let declineOOD: Float, declineInList: Float
    /// A misnamed garden plant costs curiosity, not health.
    static let identify = Utility(labelCorrect: 1, groupCorrect: 0.5, wrong: -2,
                                  declineOOD: 1, declineInList: 0)
    /// Someone may eat it. Abstention is worth far more than an answer.
    static let forage  = Utility(labelCorrect: 1, groupCorrect: 0.5, wrong: -20,
                                 declineOOD: 1, declineInList: 0)
}

/// The same 60x60 quantile grid `cascade.fit_thresholds` uses, maximising
/// expected utility under a declared out-of-list prevalence.
func gridSearch(_ cal: [(Float, Float, Bool, Bool)], utility u: Utility,
                pOOD: Float) -> (Float, Float) {
    guard !cal.isEmpty else { return (0, 0) }
    func quantiles(_ xs: [Float]) -> [Float] {
        let s = xs.sorted()
        return (0..<60).map { s[min(s.count - 1, Int(Float($0) / 59 * Float(s.count - 1))) ] }
    }
    let gGrid = quantiles(cal.map(\.1)), lGrid = quantiles(cal.map(\.0))
    let nIn = max(cal.filter(\.3).count, 1), nOut = max(cal.count - nIn, 1)
    // reweight to the declared prevalence, so an incidental calibration mix
    // cannot choose the operating point
    let wIn = (1 - pOOD) / Float(nIn), wOut = pOOD / Float(nOut)

    var best = (gGrid[0], lGrid[0]), bestU = -Float.infinity
    for tg in gGrid {
        for tl in lGrid {
            var acc: Float = 0
            for (lc, gc, correct, inList) in cal {
                let w = inList ? wIn : wOut
                if gc < tg { acc += w * (inList ? u.declineInList : u.declineOOD) }
                else if lc < tl { acc += w * (correct && inList ? u.groupCorrect : u.wrong) }
                else { acc += w * (correct && inList ? u.labelCorrect : u.wrong) }
            }
            if acc > bestU { bestU = acc; best = (tg, tl) }
        }
    }
    return best
}
