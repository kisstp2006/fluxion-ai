// SPDX-License-Identifier: BSL-1.0

//! Fluxion AI - words, pictures and video from whoever makes them, over
//! HTTP, from Zig 0.16. On every target `std.http` reaches, with nothing
//! from outside the standard library but fluxion-json.
//!
//! ```zig
//! const ai = @import("fluxion_ai");
//!
//! var client: ai.Client = .init(gpa, io, .anthropic(key));
//! defer client.deinit();
//!
//! var answer = try client.chat(.{
//!     .model = "claude-sonnet-5",
//!     .messages = &.{.user("Name three uses for a paperclip.")},
//! });
//! defer answer.deinit();
//! std.debug.print("{s}\n", .{answer.text});
//! ```
//!
//! **Three APIs, one set of types.** OpenAI's shape - which DeepSeek, xAI,
//! Groq, Mistral, OpenRouter, Together, Fireworks, Perplexity, Ollama and
//! LM Studio all speak - Anthropic's, and Google's. A `Provider` says which
//! one, where, and with what key; `Client` turns the same `ChatRequest`,
//! `ImageRequest` and `VideoRequest` into whichever the provider expects,
//! and its answers back into the same `Chat`, `Images` and `Video`.
//!
//! **Pictures and video are files here.** A picture handed to a model is
//! its bytes and a MIME type; one handed back is the PNG or JPEG the
//! provider made, byte for byte. There is no decoder in this library, for
//! pictures or for video: what it does with a file is send it, receive it
//! and name its type from the first few bytes.
//!
//! **What is not modelled is still reachable.** Every request takes an
//! `extra` JSON object whose members go into the body as they are, every
//! answer keeps its `raw` body, and `Client.call` sends anything to any
//! endpoint with the provider's key and error handling.
//!
//! **A failure says what happened.** A Zig error cannot carry a message, so
//! the provider's - "Insufficient Balance", "model not found" - is kept in
//! `Client.failure` until the next call.

const std = @import("std");

pub const Client = @import("Client.zig");
pub const Provider = @import("Provider.zig");
pub const ChatStream = @import("ChatStream.zig");

pub const Role = chat.Role;
pub const Message = chat.Message;
pub const Image = chat.Image;
pub const ChatRequest = chat.ChatRequest;
pub const Chat = chat.Chat;
pub const Finish = chat.Finish;
pub const Usage = chat.Usage;

pub const ImageRequest = image.ImageRequest;
pub const Images = image.Images;
pub const GeneratedImage = image.GeneratedImage;

pub const VideoRequest = video.VideoRequest;
pub const Video = video.Video;
pub const WaitOptions = video.WaitOptions;

pub const Model = Client.Model;
pub const Models = Client.Models;
pub const Response = Client.Response;

/// A file as bytes and a MIME type, and naming one from its signature.
pub const media = @import("media.zig");
pub const Media = media.Media;

/// The failures that are the provider's rather than the network's.
pub const Error = transport.Error;
/// What went wrong, in the provider's words. See `Client.failure`.
pub const Failure = transport.Failure;

/// The server-sent events parser the streams are read with, for a program
/// that reads some other stream.
pub const sse = @import("sse.zig");

/// fluxion-json, which every request is written with and every answer read
/// with: `ai.json.parse(gpa, answer.raw, .{})` for what a `Chat` does not carry.
pub const json = @import("fluxion_json");

const chat = @import("chat.zig");
const image = @import("image.zig");
const video = @import("video.zig");
const transport = @import("transport.zig");

test {
    _ = @import("media.zig");
    _ = @import("sse.zig");
    _ = @import("json.zig");
    _ = @import("transport.zig");
    _ = @import("Provider.zig");
    _ = @import("video.zig");
    _ = @import("api/openai.zig");
    _ = @import("api/anthropic.zig");
    _ = @import("api/gemini.zig");
    std.testing.refAllDecls(Client);
    std.testing.refAllDecls(ChatStream);
}
