// SPDX-License-Identifier: BSL-1.0

//! Asking for video.
//!
//! No provider makes a video while the request waits. Every one takes the
//! prompt, answers with an id, and makes the video in its own time, which
//! is a minute or several; the program asks after it until it is done, and
//! then fetches the file. `Client.startVideo`, `Client.videoStatus` and
//! `Client.downloadVideo` are those three steps, and `Client.waitVideo` is
//! the asking-after in a loop.
//!
//! What is fetched is the file - MP4, as every provider makes it - byte for
//! byte. Playing it, or pulling frames out of it, is not done here.

const std = @import("std");
const Io = std.Io;

const Image = @import("chat.zig").Image;

pub const VideoRequest = struct {
    /// `sora-2`, `sora-2-pro`, `veo-3.1-generate-preview`,
    /// `grok-imagine-video-1.5`...
    model: []const u8,
    prompt: []const u8,
    /// How long. Sora takes 4, 8 or 12; Veo 4, 6 or 8.
    seconds: ?u32 = null,
    /// Sora: `1280x720`, `720x1280`, `1792x1024`, `1024x1792`.
    size: ?[]const u8 = null,
    /// Veo and xAI: `16:9` or `9:16`.
    aspect_ratio: ?[]const u8 = null,
    /// Veo and xAI: `720p`, `1080p`.
    resolution: ?[]const u8 = null,
    /// Veo: what the video should not show.
    negative_prompt: ?[]const u8 = null,
    /// A picture to start from. Sora wants it the same size as the video;
    /// Veo and Sora take the file, xAI a link or the file as a data URL.
    first_frame: ?Image = null,
    /// Anything else, as the text of a JSON object. For Veo these go into
    /// `parameters` (`personGeneration`, `seed`...); for Sora's form each
    /// member becomes a field.
    extra: ?[]const u8 = null,
};

/// A video being made, or made, and the memory the answer lives in.
pub const Video = struct {
    arena: std.heap.ArenaAllocator,
    /// What to ask after it by: OpenAI's `video_...`, xAI's request id, or
    /// Gemini's operation name.
    id: []const u8 = "",
    status: Status = .queued,
    /// Percent done, where the provider says (OpenAI).
    progress: ?u8 = null,
    /// Where the finished file is, once there is one.
    url: ?[]const u8 = null,
    /// Why it failed, when it did.
    message: []const u8 = "",
    /// The response body as it came.
    raw: []const u8 = "",

    pub const Status = enum {
        queued,
        in_progress,
        completed,
        failed,

        /// The providers' words for the four states, all of them. A word
        /// this does not know is taken to mean the video is still coming,
        /// which is the answer that keeps a poll going rather than ending
        /// it wrongly.
        pub fn fromWord(word: []const u8) Status {
            const table = [_]struct { []const u8, Status }{
                .{ "queued", .queued },          .{ "pending", .queued },      .{ "submitted", .queued },
                .{ "waiting", .queued },         .{ "created", .queued },      .{ "in_progress", .in_progress },
                .{ "processing", .in_progress }, .{ "running", .in_progress }, .{ "generating", .in_progress },
                .{ "completed", .completed },    .{ "complete", .completed },  .{ "done", .completed },
                .{ "succeeded", .completed },    .{ "success", .completed },   .{ "failed", .failed },
                .{ "failure", .failed },         .{ "error", .failed },        .{ "expired", .failed },
                .{ "cancelled", .failed },       .{ "canceled", .failed },     .{ "rejected", .failed },
            };
            for (table) |entry| {
                if (std.ascii.eqlIgnoreCase(word, entry[0])) return entry[1];
            }
            return .in_progress;
        }
    };

    /// Completed or failed: asking again will not change it.
    pub fn done(video: *const Video) bool {
        return video.status == .completed or video.status == .failed;
    }

    pub fn deinit(video: *Video) void {
        video.arena.deinit();
        video.* = undefined;
    }
};

pub const WaitOptions = struct {
    /// How long between asking. OpenAI suggests ten to twenty seconds, and
    /// Google ten.
    poll_interval: Io.Duration = .fromSeconds(10),
    /// Give up after this long; null waits as long as it takes.
    timeout: ?Io.Duration = .fromSeconds(30 * 60),
};

test "Status.fromWord" {
    try std.testing.expectEqual(Video.Status.queued, Video.Status.fromWord("pending"));
    try std.testing.expectEqual(Video.Status.in_progress, Video.Status.fromWord("IN_PROGRESS"));
    try std.testing.expectEqual(Video.Status.completed, Video.Status.fromWord("done"));
    try std.testing.expectEqual(Video.Status.failed, Video.Status.fromWord("expired"));
    try std.testing.expectEqual(Video.Status.in_progress, Video.Status.fromWord("rendering_frames"));
}
