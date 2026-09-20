import CoreGraphics
import CoreVideo
import Foundation

// Everything the encoder needs that is NOT already inside its Core ML graph.
//
// Inside the graph, already done for you: the 1/255 scale, the channel mean/std,
// and the final L2 normalisation. Do not repeat any of them.
//
// Outside the graph, this file:
//   1. resize so the SHORT side is 518, preserving aspect
//   2. centre-crop 518x518
//   3. hand over BGRA pixels
//
// The resize rule is exact, not approximate. torchvision's `Resize(518)` — which
// is what produced the vectors this app's head was fitted on — computes the other
// dimension with INTEGER TRUNCATION. Rounding instead gives cosine 0.988–0.998 to
// those vectors where truncating gives 1.00000, measured over six photographs.

enum Preprocess {
    static let side = 518

    /// Resize-and-crop into a buffer the model can take.
    ///
    /// Takes a `CGImage`, deliberately, rather than a `UIImage`: a `CGImage` is
    /// the pixels as stored, with no EXIF rotation applied. That matches PIL's
    /// `Image.open()`, which is what generated `parity_fixture.json`. For photos
    /// from the camera you almost certainly *do* want the EXIF rotation applied
    /// first — see `note` at the bottom of this file.
    static func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { return nil }

        // torchvision's rule, truncating. Not `round`, not `ceil`.
        let ow: Int, oh: Int
        if w < h {
            ow = side
            oh = Int(Double(side) * Double(h) / Double(w))
        } else {
            oh = side
            ow = Int(Double(side) * Double(w) / Double(h))
        }
        let left = (ow - side) / 2
        let top = (oh - side) / 2

        var maybeBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, side, side,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary,
                                  &maybeBuffer) == kCVReturnSuccess,
              let buffer = maybeBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: side, height: side, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.interpolationQuality = .high        // bicubic-equivalent

        // NO FLIP. `CGContext.draw` already lands the image right-side-up in the
        // buffer's memory: row 0 of a CVPixelBuffer is the top row, and drawing
        // into a bitmap context accounts for that. An earlier version of this
        // file added the usual `translateBy`/`scaleBy(1, -1)` idiom and fed the
        // encoder a vertically mirrored plant — which produced a perfectly
        // plausible embedding of the wrong thing, and still classified most
        // images correctly. Measured: mean pixel error 65/255 flipped against
        // 3/255 correct. The parity fixture is what caught it.

        // Draw the whole resized image, shifted so the centre crop lands in view.
        ctx.draw(image, in: CGRect(x: CGFloat(-left), y: CGFloat(-top),
                                   width: CGFloat(ow), height: CGFloat(oh)))
        return buffer
    }
}

// note — EXIF orientation
//
// `parity_fixture.json` was generated with PIL, which does not apply EXIF
// rotation, so the fixture test must not either: pass `cgImage` straight through
// and it will match.
//
// A photo from `UIImagePickerController` or `AVCapture` usually carries an
// orientation flag, and its `cgImage` is the *unrotated* sensor data. Feeding
// that in sideways is a real accuracy loss on real photographs. Normalise it
// before calling here — draw the `UIImage` into a context once so the rotation
// is baked into the pixels, then take that image's `cgImage`.
