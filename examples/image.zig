// SPDX-License-Identifier: BSL-1.0

//! Draw a picture, save it as the file it came as, and - where it is a PNG -
//! open it with fluxion-image to see what is inside.
//!
//!     zig build image -- --provider gemini "a lighthouse in a storm, oil on canvas"
//!     zig build image -- --size 1536x1024 --count 2 "a fox in the snow"
//!     zig build image -- --file photo.png "the same scene, as a watercolour"
//!
//! The library hands back files and never looks inside them. Looking is
//! this program's business, and fluxion-image is what it looks with.

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const fluxion_image = @import("fluxion_image");
const common = @import("common.zig");

const usage =
    \\usage: zig build image -- [options] PROMPT
    \\  --size SIZE       1024x1024, 1536x1024, auto (OpenAI); 1K, 2K, 4K (Gemini)
    \\  --aspect RATIO    16:9, 1:1, ... (Gemini, Imagen, xAI)
    \\  --count N         how many pictures
    \\  --file PATH       a picture to edit or draw from; may be given more than once
;

pub fn main(init: std.process.Init) !void {
    const console: common.Console = .utf8();
    defer console.restore();
    const io = init.io;
    const arena = init.arena.allocator();

    const options = try common.parse(init, usage);
    if (options.prompt.len == 0) {
        std.debug.print("{s}\n{s}", .{ usage, common.common_usage });
        std.process.exit(2);
    }

    var client: ai.Client = .init(init.gpa, io, common.provider(init, options));
    defer client.deinit();
    const model = common.model(&client, options, .image) orelse std.process.exit(1);

    // Pictures to start from go as the files they are.
    var references: std.ArrayList(ai.Media) = .empty;
    for (options.files) |path| {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
        try references.append(arena, .fromBytes(bytes));
    }

    std.debug.print("asking {s} for {d} picture{s}...\n", .{ model, options.count, if (options.count == 1) "" else "s" });
    var result = client.generateImages(.{
        .model = model,
        .prompt = options.prompt,
        .count = options.count,
        .size = options.size,
        .aspect_ratio = options.aspect_ratio,
        .references = references.items,
    }) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer result.deinit();

    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &buffer);
    const out = &stdout.interface;

    const dir = try common.outputDir(io);
    for (result.images, 1..) |picture, n| {
        const path = try std.fmt.allocPrint(arena, "{s}/picture-{d}.{s}", .{ dir, n, picture.extension() });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = picture.bytes });
        try out.print("{s}: {s}, {d} bytes\n", .{ path, picture.mime_type, picture.bytes.len });
        if (picture.revised_prompt) |revised| try out.print("  drawn as: {s}\n", .{revised});
        try describe(init.gpa, out, picture);
    }
    if (result.text.len > 0) try out.print("the model said: {s}\n", .{result.text});
    try out.flush();
    if (result.usage.output_tokens) |n| std.debug.print("-- {d} tokens\n", .{n});
}

/// What is in the picture: its size, the colour it averages to, and how
/// much of it is see-through. PNG only, because that is what fluxion-image
/// reads; anything else stays the file it was.
fn describe(gpa: std.mem.Allocator, out: *Io.Writer, picture: ai.GeneratedImage) !void {
    if (!std.mem.eql(u8, picture.mime_type, "image/png")) {
        try out.print("  (a {s}; fluxion-image reads PNG, so this one is left as it came)\n", .{picture.mime_type});
        return;
    }
    var decoded = fluxion_image.png.decode(gpa, picture.bytes) catch |err| {
        try out.print("  fluxion-image could not read it: {t}\n", .{err});
        return;
    };
    defer decoded.deinit(gpa);

    var sum: [3]u64 = .{ 0, 0, 0 };
    var transparent: u64 = 0;
    var pixels = std.mem.window(u8, decoded.pixels, 4, 4);
    while (pixels.next()) |p| {
        for (0..3) |c| sum[c] += p[c];
        if (p[3] == 0) transparent += 1;
    }
    const count = @as(u64, decoded.width) * decoded.height;
    try out.print("  {d} by {d}, averaging #{x:0>2}{x:0>2}{x:0>2}, {d}% transparent\n", .{
        decoded.width,  decoded.height,
        sum[0] / count, sum[1] / count,
        sum[2] / count, transparent * 100 / count,
    });
}
