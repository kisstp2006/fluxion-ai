// SPDX-License-Identifier: BSL-1.0

//! Draw a test card, show it to a model that can see, and print what it
//! says it sees.
//!
//!     zig build vision -- --provider anthropic
//!     zig build vision -- --provider gemini --file photo.jpg "What is in this photo?"
//!
//! The card is arithmetic - a red disc, a yellow square, a blue ground -
//! turned into a PNG by fluxion-image, so the example needs no file to run
//! and the answer can be checked by eye. `--file` sends a picture of your
//! own instead, as the file it is.

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const fluxion_image = @import("fluxion_image");
const common = @import("common.zig");

const usage =
    \\usage: zig build vision -- [options] [QUESTION]
    \\  --file PATH       a picture to ask about, rather than the test card
;

const width = 320;
const height = 200;

pub fn main(init: std.process.Init) !void {
    const console: common.Console = .utf8();
    defer console.restore();
    const io = init.io;
    const arena = init.arena.allocator();

    const options = try common.parse(init, usage);
    var client: ai.Client = .init(init.gpa, io, common.provider(init, options));
    defer client.deinit();
    const model = common.model(&client, options, .vision) orelse std.process.exit(1);

    const picture: ai.Media = if (options.files.len > 0)
        .fromBytes(try Io.Dir.cwd().readFileAlloc(io, options.files[0], arena, .limited(32 << 20)))
    else
        .{ .bytes = try testCard(arena), .mime_type = "image/png" };
    if (options.files.len == 0) {
        const dir = try common.outputDir(io);
        const path = try std.fmt.allocPrint(arena, "{s}/test-card.png", .{dir});
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = picture.bytes });
        std.debug.print("drew {s} ({d} bytes) and am asking {s} about it...\n", .{ path, picture.bytes.len, model });
    }

    const question = if (options.prompt.len > 0)
        options.prompt
    else
        "Describe this picture exactly: which shapes are there, in which colours, and where?";

    var answer = client.chat(.{
        .model = model,
        .messages = &.{.{ .role = .user, .text = question, .images = &.{.{ .file = picture }} }},
        .max_tokens = 500,
    }) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer answer.deinit();

    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &buffer);
    try stdout.interface.print("{s}\n", .{answer.text});
    try stdout.interface.flush();
    std.debug.print("\n-- {s}, {?d} tokens in, {?d} out\n", .{ answer.model, answer.usage.input_tokens, answer.usage.output_tokens });
}

/// A blue ground, a red disc on the left, a yellow square on the right:
/// easy to describe, and easy to see whether a description is right.
fn testCard(arena: std.mem.Allocator) ![]u8 {
    const pixels = try arena.alloc(u8, width * height * 4);
    for (0..height) |y| {
        for (0..width) |x| {
            const p = pixels[(y * width + x) * 4 ..][0..4];
            const dx = @as(i64, @intCast(x)) - 90;
            const dy = @as(i64, @intCast(y)) - 100;
            const in_disc = dx * dx + dy * dy <= 55 * 55;
            const in_square = x >= 190 and x < 280 and y >= 55 and y < 145;
            p.* = if (in_disc) .{ 220, 30, 30, 255 } else if (in_square) .{ 250, 210, 20, 255 } else .{ 30, 60, 160, 255 };
        }
    }
    return fluxion_image.png.encodeAlloc(arena, .{
        .width = width,
        .height = height,
        .pixels = pixels,
        .row_pitch = width * 4,
    }, .{});
}
