import Testing
import CoreGraphics
import CoreML
import ImageIO
import Foundation
@testable import plantbook

// The real accuracy of this app, on this device, over all 618 Oregon
// photographs the bundle was measured on.
//
// It exists because the figures quoted for this model were produced on a Mac,
// and PREPROCESS_FINDINGS.md establishes that the resampling filter moves this
// encoder's embeddings more than the compute backend does — so a Mac number is a
// lower bound on the deployed cost, not the deployed cost. Only the device can
// say what the device does.
//
// DISABLED by default: it embeds 618 images at ~154 ms each, so it runs for
// two to three minutes. Delete the `.disabled` trait to run it, and read the
// result in the Report navigator.
//
// The images are gitignored. Regenerate them from narrowcast-plantid, from
// data/processed/regions/oregon plus its manifest.parquet.

private final class Anchor {}

@Suite struct AccuracyOnDevice {
    let bundle = Foundation.Bundle(for: Anchor.self)

    /// Synchronized folders flatten resources into the bundle root — which is
    /// why the six parity images had to be removed from `oregon618/`, since two
    /// copies of one filename is a build error. The subdirectory fallback stays
    /// in case that behaviour ever changes.
    func url(_ name: String) -> URL? {
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        return bundle.url(forResource: base, withExtension: ext)
            ?? bundle.url(forResource: base, withExtension: ext, subdirectory: "oregon618")
    }

    // Last run on an iPad (A16), 2026-09-20: 618 photographs, 91.4% named and
    // 100.0% of those correct, 8.6% declined, 157 ms each, 108 s total.
    @Test("accuracy over all 618 Oregon photographs",
          .disabled("~2 minutes; remove this trait to re-run"))
    func measure() throws {
        let truthURL = try #require(url("truth.json"), "truth.json is not in the test bundle")
        let truth = try JSONDecoder().decode([String: String].self,
                                             from: Data(contentsOf: truthURL))
        let cascade = try Cascade.load()
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let model = try PlantEncoder(configuration: cfg)   // once, not per image

        var named = 0, correct = 0, total = 0, missing = 0
        var declined = 0, grouped = 0
        var wrongOnes: [String] = []
        let t0 = CFAbsoluteTimeGetCurrent()

        for (file, species) in truth.sorted(by: { $0.key < $1.key }) {
            guard let u = url(file),
                  let src = CGImageSourceCreateWithURL(u as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let buf = Preprocess.pixelBuffer(from: img) else { missing += 1; continue }
            let v = widen(try model.prediction(image: buf).embedding)
            let a = cascade(v)
            total += 1
            switch a.rank {
            case .label:
                named += 1
                if a.text == species { correct += 1 }
                else if wrongOnes.count < 12 { wrongOnes.append("\(species) -> \(a.text ?? "?")") }
            case .group:   grouped += 1
            case .decline: declined += 1
            }
        }
        let secs = CFAbsoluteTimeGetCurrent() - t0

        func pct(_ n: Int, _ d: Int) -> String {
            d == 0 ? "n/a" : String(format: "%.1f%%", 100 * Double(n) / Double(d))
        }
        print("""

        ┌─ ON-DEVICE ACCURACY ─────────────────────────────────────
        │ photographs        \(total)   (missing from bundle: \(missing))
        │ names a species    \(pct(named, total))   (\(named))
        │ right when it does \(pct(correct, named))
        │ OVERALL CORRECT    \(pct(correct, total))
        │ answered at family \(pct(grouped, total))
        │ declined           \(pct(declined, total))
        │ \(String(format: "%.0f ms/photograph", secs * 1000 / Double(max(total, 1))))
        ├─ for comparison, measured on a Mac ──────────────────────
        │ torchvision bicubic (head fitted on this)   93.2% correct
        │ macOS CoreGraphics  (this pipeline, on Mac) 90.6% correct
        └──────────────────────────────────────────────────────────
        misnamed: \(wrongOnes.isEmpty ? "none" : wrongOnes.joined(separator: ", "))
        """)

        #expect(missing == 0, "\(missing) images were not in the test bundle")
        #expect(total > 500, "only \(total) photographs ran")
    }
}
