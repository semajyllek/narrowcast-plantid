import Testing
import CoreGraphics
import CoreVideo
import Foundation
@testable import plantbook

// Preprocessing, tested on images whose correct output is unambiguous.
//
// These replaced a test that compared real embeddings against a reference
// generated on a Mac. That could not work: this encoder is sensitive enough to
// the resampling filter that macOS and iOS CoreGraphics disagree by cosine 0.93
// on some photographs — while a vertically flipped image can still score 0.976.
// No threshold separates "wrong platform" from "upside down", so the reference
// approach was abandoned for one that needs no tolerance at all.
//
// Large flat colour regions, sampled well inside their boundaries, resample to
// themselves under any filter. So these assertions are exact.

private func solid(_ w: Int, _ h: Int,
                   _ fill: (_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8)) -> CGImage {
    var bytes = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let (r, g, b) = fill(x, y)
            let i = (y * w + x) * 4
            bytes[i] = r; bytes[i+1] = g; bytes[i+2] = b; bytes[i+3] = 255
        }
    }
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    let provider = CGDataProvider(data: Data(bytes) as CFData)!
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                   bytesPerRow: w * 4, space: cs, bitmapInfo: CGBitmapInfo(rawValue: info),
                   provider: provider, decode: nil, shouldInterpolate: false,
                   intent: .defaultIntent)!
}

/// (r, g, b) at a pixel of the preprocessed buffer.
private func sample(_ buf: CVPixelBuffer, _ x: Int, _ y: Int) -> (Int, Int, Int) {
    CVPixelBufferLockBaseAddress(buf, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    let base = CVPixelBufferGetBaseAddress(buf)!
    let row = base.advanced(by: y * bpr).bindMemory(to: UInt8.self, capacity: bpr)
    return (Int(row[x*4 + 2]), Int(row[x*4 + 1]), Int(row[x*4 + 0]))   // BGRA
}

@Suite struct GeometryTests {

    @Test("the image is not turned upside down")
    func noVerticalFlip() throws {
        // Top half red, bottom half blue. A flip swaps them, and this is the bug
        // that actually shipped: CGContext over a CVPixelBuffer needs no y-flip,
        // and adding the usual idiom inverted every photograph while the
        // classification test went on passing.
        let img = solid(600, 600) { _, y in y < 300 ? (255, 0, 0) : (0, 0, 255) }
        let buf = try #require(Preprocess.pixelBuffer(from: img))
        let top = sample(buf, 259, 40), bottom = sample(buf, 259, 478)
        #expect(top.0 > 200 && top.2 < 55, "top of the buffer should be RED, got \(top)")
        #expect(bottom.2 > 200 && bottom.0 < 55, "bottom should be BLUE, got \(bottom)")
    }

    @Test("the image is not mirrored")
    func noHorizontalFlip() throws {
        let img = solid(600, 600) { x, _ in x < 300 ? (0, 255, 0) : (255, 255, 255) }
        let buf = try #require(Preprocess.pixelBuffer(from: img))
        let left = sample(buf, 40, 259), right = sample(buf, 478, 259)
        #expect(left.0 < 55 && left.1 > 200, "left should be GREEN, got \(left)")
        #expect(right.0 > 200 && right.2 > 200, "right should be WHITE, got \(right)")
    }

    @Test("red stays red — the channels are not swapped")
    func noChannelSwap() throws {
        // A BGR/RGB mix-up turns this blue, and would otherwise only show up as a
        // quietly worse model.
        let img = solid(600, 600) { _, _ in (220, 30, 10) }
        let buf = try #require(Preprocess.pixelBuffer(from: img))
        let c = sample(buf, 259, 259)
        #expect(c.0 > 190 && c.1 < 70 && c.2 < 50, "expected red-ish, got \(c)")
    }

    @Test("a wide image is centre-cropped, not squashed")
    func cropsRatherThanStretches() throws {
        // 1000x500: the short side scales to 518, giving 1036x518, and the centre
        // 518 columns are kept — exactly the middle half of the original width.
        // So a green middle half survives and the red edges are cropped away.
        // Squashing instead of cropping would bring the red into view.
        let img = solid(1000, 500) { x, _ in (250...750).contains(x) ? (0, 255, 0) : (255, 0, 0) }
        let buf = try #require(Preprocess.pixelBuffer(from: img))
        for x in [20, 259, 497] {
            let c = sample(buf, x, 259)
            #expect(c.1 > 200 && c.0 < 55, "column \(x) should still be GREEN, got \(c)")
        }
    }

    @Test("a tall image is centre-cropped too")
    func cropsTallImages() throws {
        let img = solid(500, 1000) { _, y in (250...750).contains(y) ? (0, 255, 0) : (255, 0, 0) }
        let buf = try #require(Preprocess.pixelBuffer(from: img))
        for y in [20, 259, 497] {
            let c = sample(buf, 259, y)
            #expect(c.1 > 200 && c.0 < 55, "row \(y) should still be GREEN, got \(c)")
        }
    }

    @Test("the output is exactly what the model asks for")
    func outputShape() throws {
        let buf = try #require(Preprocess.pixelBuffer(from: solid(37, 900) { _, _ in (1, 2, 3) }))
        #expect(CVPixelBufferGetWidth(buf) == 518)
        #expect(CVPixelBufferGetHeight(buf) == 518)
    }
}
