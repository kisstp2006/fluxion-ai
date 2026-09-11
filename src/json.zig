// SPDX-License-Identifier: BSL-1.0

//! JSON in both directions: request bodies written straight into a buffer,
//! and answers read as `std.json.Value` and walked by a path that forgives.
//!
//! Answers are read loosely on purpose. Three APIs and a dozen servers that
//! copy one of them agree on the fields that matter and disagree on
//! everything else - a `null` here, an extra object there, a number sent as
//! a string. A walk that returns null for anything missing or misshapen, and
//! lets the caller decide what null means, survives all of them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Value = std.json.Value;
const Stringify = std.json.Stringify;

const media = @import("media.zig");

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Parse `bytes` into values that live in `arena`. Strings are copied, so
/// `bytes` may be freed afterwards.
pub fn parse(arena: Allocator, bytes: []const u8) error{ OutOfMemory, InvalidResponse }!Value {
    return std.json.parseFromSliceLeaky(Value, arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidResponse,
    };
}

/// Walk `path` from `root`: strings step into objects, integers into arrays.
/// Null as soon as a step does not fit.
///
///     at(root, .{ "choices", 0, "message", "content" })
pub fn at(root: ?Value, path: anytype) ?Value {
    var current = root orelse return null;
    inline for (path) |step| {
        const Step = @TypeOf(step);
        if (@typeInfo(Step) == .int or @typeInfo(Step) == .comptime_int) {
            const items = switch (current) {
                .array => |a| a.items,
                else => return null,
            };
            if (step >= items.len) return null;
            current = items[step];
        } else {
            const key: []const u8 = step;
            current = switch (current) {
                .object => |o| o.get(key) orelse return null,
                else => return null,
            };
        }
    }
    return current;
}

/// The string at `v`, or null for anything else - including JSON's `null`.
pub fn string(v: ?Value) ?[]const u8 {
    return switch (v orelse return null) {
        .string => |s| s,
        else => null,
    };
}

/// The whole number at `v`. Takes a float that happens to be whole, and a
/// number sent as a string, because both happen.
pub fn integer(v: ?Value) ?i64 {
    return switch (v orelse return null) {
        .integer => |i| i,
        .float => |f| if (@floor(f) == f and @abs(f) < 9.0e15) @intFromFloat(f) else null,
        .number_string, .string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        else => null,
    };
}

/// A count of tokens, or null when there is none or it makes no sense.
pub fn count(v: ?Value) ?u64 {
    const i = integer(v) orelse return null;
    return if (i < 0) null else @intCast(i);
}

pub fn boolean(v: ?Value) ?bool {
    return switch (v orelse return null) {
        .bool => |b| b,
        else => null,
    };
}

/// The items at `v`, or none when it is not an array.
pub fn array(v: ?Value) []const Value {
    return switch (v orelse return &.{}) {
        .array => |a| a.items,
        else => &.{},
    };
}

pub fn isObject(v: ?Value) bool {
    return v != null and v.? == .object;
}

/// What a provider said went wrong, from an error body in any of the shapes
/// they use: `{"error":{"message":...}}` (OpenAI, DeepSeek, Gemini, and
/// Anthropic under `"type":"error"`), `{"error":"..."}`, `{"message":...}`,
/// `{"detail":...}`, or any of those as the first item of an array, which is
/// how Gemini's streaming endpoint sends them.
pub fn errorMessage(root: Value) ?[]const u8 {
    const v = switch (root) {
        .array => |a| if (a.items.len > 0) a.items[0] else return null,
        else => root,
    };
    if (string(at(v, .{ "error", "message" }))) |m| return m;
    if (string(at(v, .{"error"}))) |m| return m;
    if (string(at(v, .{"message"}))) |m| return m;
    if (string(at(v, .{"detail"}))) |m| return m;
    if (string(at(v, .{ "detail", 0, "msg" }))) |m| return m;
    if (string(at(v, .{ "error", "type" }))) |m| return m;
    return null;
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub const ExtraError = error{
    /// `extra` is not a JSON object.
    InvalidExtra,
} || Allocator.Error;

/// The members of `extra`, a JSON object's text, or none when it is null.
pub fn parseExtra(arena: Allocator, extra: ?[]const u8) ExtraError!?std.json.ObjectMap {
    const text = extra orelse return null;
    // Numbers stay as the caller wrote them: `0.1` goes out as `0.1`, not as
    // the nearest double printed back.
    const v = std.json.parseFromSliceLeaky(Value, arena, text, .{ .parse_numbers = false }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidExtra,
    };
    return switch (v) {
        .object => |o| o,
        else => error.InvalidExtra,
    };
}

/// A request body: `std.json.Stringify` with the fields a caller put in
/// `extra` taking precedence over the ones this library writes. A field in
/// `extra` is skipped when the library comes to write its own, and written,
/// with everything else in `extra`, by `end`.
pub const Body = struct {
    s: Stringify,
    extra: ?std.json.ObjectMap,

    pub const Error = Io.Writer.Error;

    pub fn begin(w: *Io.Writer, extra: ?std.json.ObjectMap) Error!Body {
        var body: Body = .{ .s = .{ .writer = w }, .extra = extra };
        try body.s.beginObject();
        return body;
    }

    /// The members of `extra`, then the closing brace.
    pub fn end(b: *Body) Error!void {
        if (b.extra) |extra| {
            var it = extra.iterator();
            while (it.next()) |entry| {
                try b.s.objectField(entry.key_ptr.*);
                try b.s.write(entry.value_ptr.*);
            }
        }
        try b.s.endObject();
    }

    pub fn overridden(b: *const Body, name: []const u8) bool {
        const extra = b.extra orelse return false;
        return extra.contains(name);
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
        try b.s.objectField(name);
        return true;
    }

    /// Inside a nested object, where `extra` does not reach. A null optional
    /// is left out here too.
    pub fn plain(b: *Body, name: []const u8, value: anytype) Error!void {
        if (@typeInfo(@TypeOf(value)) == .optional) {
            if (value) |v| try b.plain(name, v);
            return;
        }
        try b.s.objectField(name);
        try b.write(value);
    }

    /// One value. Floats are written in plain decimal, shortest first:
    /// `0.7`, where `Stringify` would write `7e-1`.
    pub fn write(b: *Body, v: anytype) Error!void {
        switch (@typeInfo(@TypeOf(v))) {
            .float, .comptime_float => try b.s.print("{d}", .{v}),
            else => try b.s.write(v),
        }
    }

    /// `bytes` as a base64 string value, written without an intermediate copy.
    pub fn base64(b: *Body, bytes: []const u8) Error!void {
        try b.s.beginWriteRaw();
        try b.s.writer.writeByte('"');
        try media.writeBase64(b.s.writer, bytes);
        try b.s.writer.writeByte('"');
        b.s.endWriteRaw();
    }

    /// `bytes` as a `data:` URL string value.
    pub fn dataUrl(b: *Body, mime_type: []const u8, bytes: []const u8) Error!void {
        try b.s.beginWriteRaw();
        try b.s.writer.writeAll("\"data:");
        try Stringify.encodeJsonStringChars(mime_type, .{}, b.s.writer);
        try b.s.writer.writeAll(";base64,");
        try media.writeBase64(b.s.writer, bytes);
        try b.s.writer.writeByte('"');
        b.s.endWriteRaw();
    }
};

test at {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const root = try parse(arena.allocator(),
        \\{"choices":[{"message":{"content":"hi"},"n":"3","f":2.0}],"none":null}
    );
    try std.testing.expectEqualStrings("hi", string(at(root, .{ "choices", 0, "message", "content" })).?);
    try std.testing.expectEqual(@as(?i64, 3), integer(at(root, .{ "choices", 0, "n" })));
    try std.testing.expectEqual(@as(?i64, 2), integer(at(root, .{ "choices", 0, "f" })));
    try std.testing.expectEqual(null, at(root, .{ "choices", 1 }));
    try std.testing.expectEqual(null, string(at(root, .{"none"})));
    try std.testing.expectEqual(null, at(root, .{ "none", "deeper" }));
    try std.testing.expectEqual(@as(usize, 1), array(at(root, .{"choices"})).len);
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

    const extra = try parseExtra(arena.allocator(), "{\"temperature\":0.5,\"seed\":7}");
    var b: Body = try .begin(&out.writer, extra);
    try b.field("model", "m");
    try b.field("temperature", @as(f32, 1.0)); // overridden: `extra` has its own
    try b.field("top_p", @as(?f32, null)); // not set: not sent
    if (try b.key("image")) {
        try b.s.beginObject();
        try b.plain("data", "x");
        try b.s.objectField("b64");
        try b.base64("hi");
        try b.s.objectField("url");
        try b.dataUrl("image/png", "hi");
        try b.s.endObject();
    }
    try b.end();

    try std.testing.expectEqualStrings(
        \\{"model":"m","image":{"data":"x","b64":"aGk=","url":"data:image/png;base64,aGk="},"temperature":0.5,"seed":7}
    , out.written());

    try std.testing.expectError(error.InvalidExtra, parseExtra(arena.allocator(), "[1,2]"));
    try std.testing.expectError(error.InvalidExtra, parseExtra(arena.allocator(), "{broken"));
}
