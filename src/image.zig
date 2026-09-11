// SPDX-License-Identifier: BSL-1.0

//! Asking for pictures.
//!
//! What comes back is files: PNG, JPEG or WebP bytes exactly as the provider
//! made them, with the MIME type the provider gave or, failing that, the one
//! their signature says. Saving one is writing its bytes; looking inside one
//! is a decoder's job, and there is none in this library.

const std = @import("std");

const media = @import("media.zig");
const Media = media.Media;
const Usage = @import("chat.zig").Usage;

pub const ImageRequest = struct {
    /// `gpt-image-1`, `dall-e-3`, `gemini-3.1-flash-image`,
    /// `imagen-4.0-generate-001`, `grok-imagine-image-2.0`, a FLUX model at
    /// Together...
    model: []const u8,
    prompt: []const u8,
    /// How many pictures. Gemini's image models draw one per request, so
    /// more than one there is that many requests.
    count: u32 = 1,
    /// As the provider spells it: `1024x1024`, `1536x1024` or `auto` for
    /// OpenAI; `1K`, `2K` or `4K` for Gemini and Imagen.
    size: ?[]const u8 = null,
    /// `16:9`, `1:1`... for Gemini, Imagen and xAI.
    aspect_ratio: ?[]const u8 = null,
    /// `low`, `medium`, `high` or `auto` for OpenAI's GPT image models;
    /// `standard` or `hd` for DALL-E 3.
    quality: ?[]const u8 = null,
    /// Pictures to edit, or to draw from: a photo to restyle, a product to
    /// put in a scene. OpenAI takes them at `/images/edits`; Gemini's image
    /// models alongside the prompt. The files are sent as they are.
    references: []const Media = &.{},
    /// Anything else, as the text of a JSON object: `background`,
    /// `output_format`, `personGeneration`, `negativePrompt`... For Imagen
    /// these go into `parameters`; for a form (OpenAI's edits) each member
    /// becomes a field.
    extra: ?[]const u8 = null,
    /// When a provider answers with links rather than files (DALL-E, xAI,
    /// Together by default), fetch them, so that `bytes` is filled either
    /// way. The link is fetched without the key: it is somebody's storage,
    /// not the provider's API.
    fetch_urls: bool = true,
};

pub const GeneratedImage = struct {
    /// The file as the provider sent it. Empty only when the provider
    /// answered with a link and `fetch_urls` was off.
    bytes: []const u8 = "",
    /// `image/png`, `image/jpeg`, `image/webp`.
    mime_type: []const u8 = "",
    /// Where the provider put it, when it answered with a link. Such links
    /// usually stop working within the hour.
    url: ?[]const u8 = null,
    /// The prompt as the provider rewrote it before drawing, where it says.
    revised_prompt: ?[]const u8 = null,

    pub fn file(image: GeneratedImage) Media {
        return .{ .bytes = image.bytes, .mime_type = image.mime_type };
    }

    /// `png`, `jpg`, `webp`: what to call the file.
    pub fn extension(image: GeneratedImage) []const u8 {
        return media.extensionOf(image.mime_type);
    }
};

/// Pictures, and the memory they live in.
pub const Images = struct {
    arena: std.heap.ArenaAllocator,
    images: []const GeneratedImage = &.{},
    /// What the model said alongside, where it says anything (Gemini).
    text: []const u8 = "",
    usage: Usage = .{},
    /// The last response body as it came, base64 and all.
    raw: []const u8 = "",

    pub fn deinit(images: *Images) void {
        images.arena.deinit();
        images.* = undefined;
    }
};
