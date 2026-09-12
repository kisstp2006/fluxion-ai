// SPDX-License-Identifier: BSL-1.0

//! JSON in both directions, through fluxion-json: request bodies written
//! with its `Writer` straight into a buffer, and answers read as its `Value`
//! trees, whose lookups forgive.
//!
//! Answers are read loosely on purpose. Three APIs and a dozen servers that
//! copy one of them agree on the fields that matter and disagree on
//! everything else - a `null` here, an extra object there, a number sent as
//! a string. A walk that returns null for anything missing or misshapen, and
//! lets the caller decide what null means, survives all of them - and in
//! fluxion-json every lookup is one: `root.at("/choices/0/message")` is
//! `.null` as soon as a step does not fit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const fluxion_json = @import("fluxion_json");
const Value = fluxion_json.Value;
const Object = fluxion_json.Object;
const Writer = fluxion_json.Writer;

const media = @import("media.zig");

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Parse `bytes` into a tree that lives in `arena`. Strings are copied, so
/// `bytes` may be freed afterwards.
pub fn parse(arena: Allocator, bytes: []const u8) error{ OutOfMemory, InvalidResponse }!Value {
    // Never deinit: the document is in `arena`, and goes with it.
    const doc = fluxion_json.parse(arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidResponse,
    };
    return doc.root;
}

/// The whole number at `v`. Takes a float that happens to be whole, and a
/// number sent as a string, because both happen.
pub fn integer(v: Value) ?i64 {
    if (v.asInt(i64)) |i| return i;
    return std.fmt.parseInt(i64, v.asString() orelse return null, 10) catch null;
}

/// A count of tokens, or null when there is none or it makes no sense.
pub fn count(v: Value) ?u64 {
    return std.math.cast(u64, integer(v) orelse return null);
}

/// What a provider said went wrong, from an error body in any of the shapes
/// they use: `{"error":{"message":...}}` (OpenAI, DeepSeek, Gemini, and
/// Anthropic under `"type":"error"`), `{"error":"..."}`, `{"message":...}`,
/// `{"detail":...}`, or any of those as the first item of an array, which is
/// how Gemini's streaming endpoint sends them.
pub fn errorMessage(root: Value) ?[]const u8 {
    const v = if (root == .array) root.get(0) else root;
    return v.at("/error/message").asString() orelse
        v.get("error").asString() orelse
        v.get("message").asString() orelse
        v.get("detail").asString() orelse
        v.at("/detail/0/msg").asString() orelse
        v.at("/error/type").asString();
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub const ExtraError = error{
    /// `extra` is not a JSON object.
    InvalidExtra,
} || Allocator.Error;

/// The members of `extra`, a JSON object's text, in `arena`; none when it is
/// null.
pub fn parseExtra(arena: Allocator, extra: ?[]const u8) ExtraError!?*Object {
    const text = extra orelse return null;
    // Numbers go out again as the same numbers - `0.1` as `0.1` - except an
    // integer past 64 bits, which comes back a float.
    const doc = fluxion_json.parse(arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidExtra,
    };
    return doc.root.asObject() orelse error.InvalidExtra;
}

/// The text of a request body, in `a`: `write(&body, args...)` between the
/// braces, with `extra` merged in. `gpa` holds what is needed only while it
/// is written.
pub fn bodyText(a: Allocator, gpa: Allocator, extra: ?*Object, comptime write: anytype, args: anytype) Body.Error![]const u8 {
    // Written twice, to count and then into exactly that much of `a`: a body
    // with a picture in it is megabytes, and grown in an arena it would leave
    // every smaller copy behind.
    var counter: Io.Writer.Discarding = .init(&.{});
    try writeBody(gpa, &counter.writer, extra, write, args);
    const text = try a.alloc(u8, @intCast(counter.fullCount()));
    var out: Io.Writer = .fixed(text);
    try writeBody(gpa, &out, extra, write, args);
    std.debug.assert(out.end == text.len);
    return text;
}

fn writeBody(gpa: Allocator, out: *Io.Writer, extra: ?*Object, comptime write: anytype, args: anytype) Body.Error!void {
    var writer: Writer = .init(out, .{});
    var body: Body = try .begin(gpa, &writer, extra);
    try @call(.auto, write, .{&body} ++ args);
    try body.end();
}

/// A request body: an object on a fluxion-json `Writer`, with the fields a
/// caller put in `extra` taking precedence over the ones this library
/// writes. A field in `extra` is skipped when the library comes to write its
/// own, and written, with everything else in `extra`, by `end`.
pub const Body = struct {
    w: *Writer,
    extra: ?*Object,
    /// Holds a picture's base64 text while it is written.
    gpa: Allocator,

    pub const Error = Writer.Error;

    pub fn begin(gpa: Allocator, w: *Writer, extra: ?*Object) Error!Body {
        try w.beginObject();
        return .{ .w = w, .extra = extra, .gpa = gpa };
    }

    /// The members of `extra`, then the closing brace.
    pub fn end(b: *Body) Error!void {
        if (b.extra) |extra| {
            for (extra.keys(), extra.values()) |name, value| try b.w.field(name, value);
        }
        try b.w.endObject();
    }

    pub fn overridden(b: *const Body, name: []const u8) bool {
        const extra = b.extra orelse return false;
        return extra.has(name);
    }

    /// `"name": value`, unless `extra` has its own, or `value` is a null
    /// optional - a field that was not set is a field that is not sent.
    pub fn field(b: *Body, name: []const u8, value: anytype) Error!void {
        if (b.overridden(name)) return;
        try b.plain(name, value);
    }

    /// The key of a field whose value the caller writes next - an object or
    /// an array. False, and nothing written, when `extra` has its own.
    pub fn key(b: *Body, name: []const u8) Error!bool {
        if (b.overridden(name)) return false;
        try b.w.key(name);
        return true;
    }

    /// Inside a nested object, where `extra` does not reach. A null optional
    /// is left out here too.
    pub fn plain(b: *Body, name: []const u8, value: anytype) Error!void {
        if (@typeInfo(@TypeOf(value)) == .optional) {
            if (value) |v| try b.plain(name, v);
            return;
        }
        try b.w.field(name, value);
    }

    /// `bytes` as a base64 string value.
    pub fn base64(b: *Body, bytes: []const u8) Error!void {
        try b.encoded(&.{}, bytes);
    }

    /// `bytes` as a `data:` URL string value.
    pub fn dataUrl(b: *Body, mime_type: []const u8, bytes: []const u8) Error!void {
        try b.encoded(&.{ "data:", mime_type, ";base64," }, bytes);
    }

    /// `prefix`, then `bytes` in base64, as one string value. fluxion-json
    /// takes a string whole, so the text is put together first, and let go
    /// as soon as it is written.
    fn encoded(b: *Body, prefix: []const []const u8, bytes: []const u8) Error!void {
        var len = std.base64.standard.Encoder.calcSize(bytes.len);
        for (prefix) |part| len += part.len;
        var text: Io.Writer.Allocating = try .initCapacity(b.gpa, len);
        defer text.deinit();
        for (prefix) |part| try text.writer.writeAll(part);
        try media.writeBase64(&text.writer, bytes);
        try b.w.writeString(text.written());
    }
};

test integer {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const root = try parse(arena.allocator(),
        \\{"n":"3","f":2.0,"i":-7,"half":2.5,"none":null,"word":"x"}
    );
    try std.testing.expectEqual(@as(?i64, 3), integer(root.get("n")));
    try std.testing.expectEqual(@as(?i64, 2), integer(root.get("f")));
    try std.testing.expectEqual(@as(?i64, -7), integer(root.get("i")));
    try std.testing.expectEqual(null, integer(root.get("half")));
    try std.testing.expectEqual(null, integer(root.get("none")));
    try std.testing.expectEqual(null, integer(root.get("word")));
    try std.testing.expectEqual(null, integer(root.get("missing")));
    try std.testing.expectEqual(@as(?u64, 3), count(root.get("n")));
    try std.testing.expectEqual(null, count(root.get("i")));
    try std.testing.expectError(error.InvalidResponse, parse(arena.allocator(), "{\"choices\":"));
}

test errorMessage {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"error\":{\"message\":\"Incorrect API key\",\"type\":\"invalid_request_error\"}}", "Incorrect API key" },
        .{ "{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}", "invalid x-api-key" },
        .{ "[{\"error\":{\"code\":400,\"message\":\"API key not valid\",\"status\":\"INVALID_ARGUMENT\"}}]", "API key not valid" },
        .{ "{\"error\":\"model not found\"}", "model not found" },
        .{ "{\"detail\":[{\"msg\":\"field required\"}]}", "field required" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], errorMessage(try parse(a, c[0])).?);
}

test Body {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var w: Writer = .init(&out.writer, .{});

    const extra = try parseExtra(arena.allocator(), "{\"temperature\":0.5,\"seed\":7}");
    var b: Body = try .begin(gpa, &w, extra);
    try b.field("model", "m");
    try b.field("temperature", @as(f32, 1.0)); // overridden: `extra` has its own
    try b.field("top_p", @as(?f32, null)); // not set: not sent
    if (try b.key("image")) {
        try b.w.beginObject();
        try b.plain("data", "x");
        try b.w.key("b64");
        try b.base64("hi");
        try b.w.key("url");
        try b.dataUrl("image/png", "hi");
        try b.w.endObject();
    }
    try b.end();

    try std.testing.expectEqualStrings(
        \\{"model":"m","image":{"data":"x","b64":"aGk=","url":"data:image/png;base64,aGk="},"temperature":0.5,"seed":7}
    , out.written());

    try std.testing.expectError(error.InvalidExtra, parseExtra(arena.allocator(), "[1,2]"));
    try std.testing.expectError(error.InvalidExtra, parseExtra(arena.allocator(), "{broken"));
}

test bodyText {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try bodyText(a, std.testing.allocator, try parseExtra(a, "{\"n\":2}"), struct {
        fn write(b: *Body, picture: []const u8) Body.Error!void {
            try b.field("model", "m");
            try b.w.key("data");
            try b.base64(picture);
        }
    }.write, .{"\x89PNG\r\n\x1a\n"});
    try std.testing.expectEqualStrings("{\"model\":\"m\",\"data\":\"iVBORw0KGgo=\",\"n\":2}", text);
}
