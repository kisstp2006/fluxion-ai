// SPDX-License-Identifier: BSL-1.0

//! Asking for words: a conversation going in, an answer coming out.

const std = @import("std");

const Media = @import("media.zig").Media;
const GeneratedImage = @import("image.zig").GeneratedImage;

pub const Role = enum {
    /// Instructions that frame the whole conversation. Anthropic and Gemini
    /// keep these apart from the turns; they are moved there on the way out.
    system,
    user,
    assistant,
};

/// One turn of a conversation.
pub const Message = struct {
    role: Role,
    /// What was said.
    text: []const u8 = "",
    /// Pictures that go with it, for a model that can see. Sent before the
    /// text, which is the order the providers recommend.
    images: []const Image = &.{},

    pub fn system(text: []const u8) Message {
        return .{ .role = .system, .text = text };
    }

    pub fn user(text: []const u8) Message {
        return .{ .role = .user, .text = text };
    }

    pub fn assistant(text: []const u8) Message {
        return .{ .role = .assistant, .text = text };
    }
};

/// A picture given to a model to look at.
pub const Image = union(enum) {
    /// The file's own bytes - PNG, JPEG, WebP, GIF - sent base64-encoded.
    /// Nothing here decodes it: the provider reads the file.
    file: Media,
    /// A link the provider fetches for itself. OpenAI and Anthropic take any
    /// public URL; Gemini takes the URIs of its own File API.
    url: []const u8,

    /// `bytes`, with the type read from their signature.
    pub fn fromBytes(bytes: []const u8) Image {
        return .{ .file = .fromBytes(bytes) };
    }

    pub fn fromUrl(url: []const u8) Image {
        return .{ .url = url };
    }
};

pub const ChatRequest = struct {
    /// As the provider spells it: `gpt-5-mini`, `claude-sonnet-5`,
    /// `deepseek-flash`, `gemini-3.5-flash`, `llama3.2`.
    model: []const u8,
    messages: []const Message,
    /// Instructions ahead of everything. Messages with `.system` are added
    /// after it.
    system: ?[]const u8 = null,
    /// The longest answer, in tokens. Anthropic will not answer without one,
    /// so 4096 is sent there when this is null; everyone else uses their own
    /// default.
    max_tokens: ?u32 = null,
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    /// Stop as soon as one of these is written.
    stop: []const []const u8 = &.{},
    /// Anything else, as the text of a JSON object whose members go into
    /// the request as they are: `reasoning_effort`, `response_format`,
    /// `thinking`, `tools`, `safetySettings`... A member here replaces a
    /// field of the same name this library would have written.
    extra: ?[]const u8 = null,
};

/// Why an answer ended.
pub const Finish = enum {
    /// It was finished, or it wrote one of the stop sequences.
    stop,
    /// It ran into `max_tokens`, or the end of the context window.
    length,
    /// A filter stopped it, or the model refused.
    content_filter,
    /// It stopped to call a tool; the call is in `raw`.
    tool_use,
    /// A reason this library has no name for. `finish_reason` has the word.
    other,
    /// No reason was given.
    unknown,
};

pub const Usage = struct {
    /// Tokens read: the prompt, the history, the pictures - including any
    /// the provider served from a cache.
    input_tokens: ?u64 = null,
    /// Tokens written: the answer, and the thinking where it is billed.
    output_tokens: ?u64 = null,
};

/// An answer, and the memory it lives in.
pub const Chat = struct {
    arena: std.heap.ArenaAllocator,
    /// The answer.
    text: []const u8 = "",
    /// The thinking that came before it, where a provider shows it:
    /// DeepSeek's `reasoning_content`, Claude's thinking blocks, Gemini's
    /// thought summaries, OpenRouter's `reasoning`. Empty otherwise.
    reasoning: []const u8 = "",
    /// Pictures in the answer, from the models that draw in conversation:
    /// Gemini's image models, and the ones OpenRouter serves the same way.
    images: []const GeneratedImage = &.{},
    finish: Finish = .unknown,
    /// Why it ended, in the provider's own word: `stop`, `end_turn`,
    /// `MAX_TOKENS`.
    finish_reason: []const u8 = "",
    usage: Usage = .{},
    /// The model that answered, which can be more exact than the one asked.
    model: []const u8 = "",
    id: []const u8 = "",
    /// The response body as it came. Tool calls, citations, log
    /// probabilities: everything this struct does not carry is in here, to
    /// be read with `std.json`.
    raw: []const u8 = "",

    pub fn deinit(chat: *Chat) void {
        chat.arena.deinit();
        chat.* = undefined;
    }
};
