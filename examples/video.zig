// SPDX-License-Identifier: BSL-1.0

//! Make a video, wait for it, and save it.
//!
//!     zig build video -- --seconds 4 "a paper boat drifting down a rainy street"
//!     zig build video -- --provider gemini --aspect 16:9 "waves at sunset, slow pan"
//!     zig build video -- --file first-frame.png "the camera slowly pulls back"
//!
//! A video takes minutes. The provider takes the job and answers at once;
//! this program asks after it every ten seconds, printing how far along it
//! is, and fetches the file when it is done - as the MP4 it is, into
//! `zig-out/video.mp4`.
//!
//! Sora wants a first frame exactly the size of the video. The frame's size
//! is read from its PNG header with fluxion-image, and asked for.

const std = @import("std");
const Io = std.Io;
const ai = @import("fluxion_ai");
const fluxion_image = @import("fluxion_image");
const common = @import("common.zig");

const usage =
    \\usage: zig build video -- [options] PROMPT
    \\  --seconds N       how long: 4, 8, 12 (Sora); 4, 6, 8 (Veo)
    \\  --size WxH        1280x720, 720x1280, ... (Sora)
    \\  --aspect RATIO    16:9 or 9:16 (Veo, xAI)
    \\  --file PATH       a picture to start from
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
    const model = common.model(&client, options, .video) orelse std.process.exit(1);

    var size = options.size;
    const first_frame: ?ai.Image = if (options.files.len > 0) frame: {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, options.files[0], arena, .limited(32 << 20));
        if (size == null and client.provider.api == .openai) {
            if (fluxion_image.png.size(bytes)) |frame_size| {
                size = try std.fmt.allocPrint(arena, "{d}x{d}", .{ frame_size.width, frame_size.height });
                std.debug.print("the first frame is {s}; asking for a video that size\n", .{size.?});
            } else |_| {}
        }
        break :frame .fromBytes(bytes);
    } else null;

    var started = client.startVideo(.{
        .model = model,
        .prompt = options.prompt,
        .seconds = options.seconds,
        .size = size,
        .aspect_ratio = options.aspect_ratio,
        .first_frame = first_frame,
    }) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    defer started.deinit();
    std.debug.print("{s} took the job: {s}\n", .{ model, started.id });

    // Ask after it until it is done, saying how it is going.
    const start = Io.Clock.awake.now(io);
    var video = while (true) {
        var status = client.videoStatus(started.id) catch |err| {
            common.explain(&client, err);
            std.process.exit(1);
        };
        const elapsed = start.untilNow(io, .awake).toSeconds();
        if (status.progress) |p| {
            std.debug.print("  {d:>4}s  {t}, {d}%\n", .{ elapsed, status.status, p });
        } else {
            std.debug.print("  {d:>4}s  {t}\n", .{ elapsed, status.status });
        }
        if (status.done()) break status;
        status.deinit();
        try io.sleep(.fromSeconds(10), .awake);
    };
    defer video.deinit();

    if (video.status == .failed) {
        std.debug.print("the video failed: {s}\n", .{if (video.message.len > 0) video.message else "no reason given"});
        std.process.exit(1);
    }

    const dir = try common.outputDir(io);
    const path = try std.fmt.allocPrint(arena, "{s}/video.mp4", .{dir});
    var file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    const n = client.downloadVideo(&video, &writer.interface) catch |err| {
        common.explain(&client, err);
        std.process.exit(1);
    };
    try writer.interface.flush();
    std.debug.print("saved {s}: {d} bytes\n", .{ path, n });
}
