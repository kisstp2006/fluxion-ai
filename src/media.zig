// SPDX-License-Identifier: BSL-1.0

//! What kind of file a run of bytes is, told from the first few of them.
//!
//! Every picture and video format a provider sends or takes begins with a
//! signature of its own, and the signature is all that is read here. Nothing
//! in this file, or anywhere in this library, decodes a picture or a video:
//! what comes back from a provider is handed on as the file it is, and what
//! goes to one is sent as the file it was. A program that wants the pixels
//! brings its own decoder.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// A file, as its bytes and its MIME type. What a picture or a video is on
/// its way to a provider, and on its way back.
pub const Media = struct {
    bytes: []const u8,
    /// `image/png`, `image/jpeg`, `video/mp4`, ...
    mime_type: []const u8,

    /// `bytes`, with the type read from their signature, or
    /// `application/octet-stream` when it is none this knows.
    pub fn fromBytes(bytes: []const u8) Media {
        return .{ .bytes = bytes, .mime_type = sniff(bytes) orelse "application/octet-stream" };
    }

    /// The usual extension for this kind of file, without the dot. What to
    /// call it when it is saved.
    pub fn extension(media: Media) []const u8 {
        return extensionOf(media.mime_type);
    }
};

/// The MIME type of `bytes`, from the signature at their start, or null when
/// it is not one of the formats a provider sends or takes.
pub fn sniff(bytes: []const u8) ?[]const u8 {
    const startsWith = std.mem.startsWith;
    if (startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) return "image/png";
    if (startsWith(u8, bytes, "\xff\xd8\xff")) return "image/jpeg";
    if (startsWith(u8, bytes, "GIF87a") or startsWith(u8, bytes, "GIF89a")) return "image/gif";
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "WEBP"))
        return "image/webp";
    if (startsWith(u8, bytes, "\x1a\x45\xdf\xa3")) return "video/webm";
    // ISO base media: a size, then `ftyp` and a brand that says which.
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[4..8], "ftyp")) {
        const brand = bytes[8..12];
        for ([_][]const u8{ "avif", "avis" }) |b| if (std.mem.eql(u8, brand, b)) return "image/avif";
        for ([_][]const u8{ "heic", "heix", "heim", "heis", "hevc", "mif1" }) |b|
            if (std.mem.eql(u8, brand, b)) return "image/heic";
        if (std.mem.eql(u8, brand, "qt  ")) return "video/quicktime";
        return "video/mp4";
    }
    return null;
}

/// The extension a file of `mime_type` is usually saved under, without the
/// dot; `bin` for a type this does not know.
pub fn extensionOf(mime_type: []const u8) []const u8 {
    const table = [_]struct { []const u8, []const u8 }{
        .{ "image/png", "png" },
        .{ "image/jpeg", "jpg" },
        .{ "image/jpg", "jpg" },
        .{ "image/gif", "gif" },
        .{ "image/webp", "webp" },
        .{ "image/avif", "avif" },
        .{ "image/heic", "heic" },
        .{ "video/mp4", "mp4" },
        .{ "video/quicktime", "mov" },
        .{ "video/webm", "webm" },
    };
    // Parameters (`; charset=...`) are not part of the type.
    const bare = std.mem.trim(u8, mime_type[0 .. std.mem.findScalar(u8, mime_type, ';') orelse mime_type.len], " ");
    for (table) |entry| {
        if (std.ascii.eqlIgnoreCase(bare, entry[0])) return entry[1];
    }
    return "bin";
}

/// Write `bytes` base64-encoded, a few kilobytes at a time rather than three
/// bytes at a time: a picture is megabytes of it.
pub fn writeBase64(w: *Io.Writer, bytes: []const u8) Io.Writer.Error!void {
    const encoder = std.base64.standard.Encoder;
    var out: [4096]u8 = undefined;
    var rest = bytes;
    while (rest.len > 0) {
        // 3072 bytes in, exactly 4096 characters out, no padding until the end.
        const n = @min(rest.len, out.len / 4 * 3);
        try w.writeAll(encoder.encode(&out, rest[0..n]));
        rest = rest[n..];
    }
}

pub const Base64Error = error{InvalidBase64} || Allocator.Error;

/// Decode base64 the way providers actually write it: the standard alphabet
/// or the URL-safe one, padded or not, and forgiving of line breaks.
pub fn decodeBase64(gpa: Allocator, text: []const u8) Base64Error![]u8 {
    var compact: []const u8 = std.mem.trim(u8, text, " \t\r\n");
    var owned: ?[]u8 = null;
    defer if (owned) |o| gpa.free(o);
    if (std.mem.findAny(u8, compact, " \t\r\n") != null) {
        const o = try gpa.alloc(u8, compact.len);
        owned = o;
        var n: usize = 0;
        for (compact) |c| switch (c) {
            ' ', '\t', '\r', '\n' => {},
            else => {
                o[n] = c;
                n += 1;
            },
        };
        compact = o[0..n];
    }

    const url_safe = std.mem.findAny(u8, compact, "-_") != null;
    const padded = compact.len % 4 == 0;
    const decoder = switch (url_safe) {
        true => if (padded) std.base64.url_safe.Decoder else std.base64.url_safe_no_pad.Decoder,
        false => if (padded) std.base64.standard.Decoder else std.base64.standard_no_pad.Decoder,
    };
    const len = decoder.calcSizeForSlice(compact) catch return error.InvalidBase64;
    const out = try gpa.alloc(u8, len);
    errdefer gpa.free(out);
    decoder.decode(out, compact) catch return error.InvalidBase64;
    return out;
}

/// The two halves of a `data:<type>;base64,<data>` URL, which is how the
/// OpenAI-shaped APIs pass pictures inline.
pub const DataUrl = struct {
    mime_type: []const u8,
    base64: []const u8,

    /// Null when `url` is not a base64 data URL.
    pub fn parse(url: []const u8) ?DataUrl {
        if (!std.mem.startsWith(u8, url, "data:")) return null;
        const comma = std.mem.findScalar(u8, url, ',') orelse return null;
        const header = url[5..comma];
        if (!std.mem.endsWith(u8, header, ";base64")) return null;
        const mime_type = header[0 .. header.len - ";base64".len];
        return .{
            .mime_type = if (mime_type.len == 0) "application/octet-stream" else mime_type,
            .base64 = url[comma + 1 ..],
        };
    }
};

test sniff {
    try std.testing.expectEqualStrings("image/png", sniff("\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").?);
    try std.testing.expectEqualStrings("image/jpeg", sniff("\xff\xd8\xff\xe0\x00\x10JFIF").?);
    try std.testing.expectEqualStrings("image/gif", sniff("GIF89a\x01\x00").?);
    try std.testing.expectEqualStrings("image/webp", sniff("RIFF\x24\x00\x00\x00WEBPVP8 ").?);
    try std.testing.expectEqualStrings("video/mp4", sniff("\x00\x00\x00\x20ftypisom\x00\x00\x02\x00").?);
    try std.testing.expectEqualStrings("video/quicktime", sniff("\x00\x00\x00\x14ftypqt  ").?);
    try std.testing.expectEqualStrings("image/avif", sniff("\x00\x00\x00\x1cftypavif").?);
    try std.testing.expectEqualStrings("video/webm", sniff("\x1a\x45\xdf\xa3\x9f").?);
    try std.testing.expectEqual(null, sniff("hello"));
    try std.testing.expectEqual(null, sniff(""));
}

test extensionOf {
    try std.testing.expectEqualStrings("png", extensionOf("image/png"));
    try std.testing.expectEqualStrings("jpg", extensionOf("IMAGE/JPEG"));
    try std.testing.expectEqualStrings("mp4", extensionOf("video/mp4; codecs=avc1"));
    try std.testing.expectEqualStrings("bin", extensionOf("application/x-unknown"));
}

test "base64 both ways" {
    const gpa = std.testing.allocator;
    var sample: [10_000]u8 = undefined;
    for (&sample, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try writeBase64(&out.writer, &sample);

    var expected: [std.base64.standard.Encoder.calcSize(sample.len)]u8 = undefined;
    try std.testing.expectEqualStrings(std.base64.standard.Encoder.encode(&expected, &sample), out.written());

    const back = try decodeBase64(gpa, out.written());
    defer gpa.free(back);
    try std.testing.expectEqualSlices(u8, &sample, back);

    // Unpadded, URL-safe, and broken across lines, as some servers send it.
    const odd = try decodeBase64(gpa, "_-8\r\n");
    defer gpa.free(odd);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0xef }, odd);

    try std.testing.expectError(error.InvalidBase64, decodeBase64(gpa, "*not base64*"));
}

test DataUrl {
    const d = DataUrl.parse("data:image/png;base64,iVBORw0K").?;
    try std.testing.expectEqualStrings("image/png", d.mime_type);
    try std.testing.expectEqualStrings("iVBORw0K", d.base64);
    try std.testing.expectEqual(null, DataUrl.parse("https://example.com/a.png"));
    try std.testing.expectEqual(null, DataUrl.parse("data:text/plain,hello"));
}
