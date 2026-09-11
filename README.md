# Fluxion AI

Words, pictures and video from whoever makes them. For Zig 0.16.

| Piece | What it is |
| --- | --- |
| `Client` | One provider, one connection pool, and every call: `chat`, `stream`, `generateImages`, `startVideo` / `videoStatus` / `waitVideo` / `downloadVideo`, `listModels`, `download`, `call`. |
| `Provider` | Who is asked: an API shape, a base URL, a key. Presets for OpenAI, Anthropic, DeepSeek, Gemini, xAI, Groq, Mistral, OpenRouter, Together, Fireworks, Perplexity, Ollama and LM Studio; a literal for anyone else. |
| `ChatStream` | An answer arriving a few words at a time. |
| `Message`, `Image`, `ChatRequest`, `Chat` | A conversation going in - with pictures, for a model that can see - and an answer coming out. |
| `ImageRequest`, `Images` | Pictures asked for, and the files that come back. |
| `VideoRequest`, `Video` | A video asked for, the job while it is made, and where the file is. |
| `Failure` | What went wrong, in the provider's own words. |
| `media` | A file as bytes and a MIME type, named from its first few bytes. |
| `sse` | The server-sent events parser the streams are read with. |

```zig
const ai = @import("fluxion_ai");

var client: ai.Client = .init(gpa, io, .deepseek(key));
defer client.deinit();

var answer = try client.chat(.{
    .model = "deepseek-flash",
    .system = "Answer in one sentence.",
    .messages = &.{.user("Why is the sky blue?")},
});
defer answer.deinit();
std.debug.print("{s}\n", .{answer.text});
```

Nothing from outside the standard library: `std.http.Client` does the
talking, over `std.crypto.tls` with the system's certificates, so it runs
wherever Zig's standard library reaches a network - Windows, Linux, macOS,
the BSDs, on x86 and ARM alike.

## Three shapes, one set of types

Almost every provider speaks one of three APIs. **OpenAI's** is the one the
others copy: DeepSeek, xAI, Groq, Mistral, OpenRouter, Together, Fireworks,
Perplexity, and the servers that run a model on this machine - Ollama, LM
Studio, vLLM, llama.cpp - all take the same chat completions. **Anthropic's**
is the second, and **Google's** the third. A `Provider` says which one, where
and with what key; the `Client` turns the same `ChatRequest` into whichever
the provider expects, and each answer back into the same `Chat`.

```zig
client.provider = .anthropic(anthropic_key);   // same client, same connections
client.provider = .preset(.ollama, null);      // a model on this machine
client.provider = .compatible("http://gpu-box:8000/v1", null);  // anything else OpenAI-shaped
client.provider = try .fromEnvironment(.gemini, init.environ_map);  // GEMINI_API_KEY
```

| Provider | Shape | Words | Sees | Draws | Films | Key from |
| --- | --- | :-: | :-: | :-: | :-: | --- |
| `openai` | OpenAI | yes | yes | yes | yes (Sora) | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | yes | yes | - | - | `ANTHROPIC_API_KEY` |
| `deepseek` | OpenAI | yes | - | - | - | `DEEPSEEK_API_KEY` |
| `gemini` | Google | yes | yes | yes (Gemini, Imagen) | yes (Veo) | `GEMINI_API_KEY` |
| `xai` | OpenAI | yes | yes | yes | yes | `XAI_API_KEY` |
| `together` | OpenAI | yes | yes | yes | - | `TOGETHER_API_KEY` |
| `openrouter` | OpenAI | yes | yes | in chat | - | `OPENROUTER_API_KEY` |
| `groq`, `mistral`, `fireworks`, `perplexity` | OpenAI | yes | model by model | - | - | `GROQ_API_KEY`, ... |
| `ollama`, `lm_studio` | OpenAI | yes | model by model | - | - | none |

"Sees" and "draws" depend on the model as much as the provider; a column
says what the provider's API can carry. What DeepSeek calls
`reasoning_content`, what Claude calls thinking and Gemini thought
summaries all arrive as `Chat.reasoning`.

**The copies are not quite copies.** OpenAI's reasoning models refuse
`max_tokens` and want `max_completion_tokens`; most servers that copy the API
know only the older name. Some send token counts at the end of a stream only
when asked, and some answer the asking with a 400. These are fields of
`Provider` - `max_tokens_field`, `stream_usage` - and the presets set them.

## Streaming

```zig
const stream = try client.stream(.{
    .model = "claude-sonnet-5",
    .messages = &.{.user("Write a haiku about Zig.")},
});
defer stream.deinit();

while (try stream.next()) |event| switch (event) {
    .text => |words| try out.writeAll(words),
    .reasoning => |thought| try err.writeAll(thought),
    .image => |picture| try save(picture),
};
// stream.text.items is the whole answer; stream.usage what it cost.
```

Three APIs stream three ways - OpenAI's list of deltas ending in `[DONE]`,
Anthropic's named events, Google's whole responses one after another - and
all of them come out of `next` as the same three kinds of event.

## Pictures and video are files

**A picture going in is its bytes.** `Image.fromBytes(png)` names the type
from the file's signature and sends it base64-encoded; `Image.fromUrl` sends
a link for the provider to fetch.

```zig
var answer = try client.chat(.{
    .model = "gemini-3.5-flash",
    .messages = &.{.{
        .role = .user,
        .text = "What is in this picture?",
        .images = &.{.fromBytes(photo_bytes)},
    }},
});
```

**A picture coming out is the file the provider made**, byte for byte:

```zig
var result = try client.generateImages(.{
    .model = "gpt-image-1",
    .prompt = "a lighthouse in a storm, oil on canvas",
    .size = "1536x1024",
});
defer result.deinit();
for (result.images) |picture| {
    // picture.bytes is a PNG, picture.mime_type says so.
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "lighthouse.png", .data = picture.bytes });
}
```

Where a provider answers with a link instead - DALL-E, xAI, Together - the
link is fetched, so that `bytes` is filled either way. `references` sends
pictures to edit or to draw from: OpenAI's `/images/edits` as a form,
Gemini's image models alongside the prompt.

**A video is a job.** No provider makes one while the request waits:

```zig
var job = try client.startVideo(.{ .model = "sora-2", .prompt = "waves at sunset", .seconds = 8 });
defer job.deinit();

var video = try client.waitVideo(job.id, .{});   // asks every ten seconds
defer video.deinit();

_ = try client.downloadVideo(&video, &file_writer.interface);   // the MP4, into a file
```

`videoStatus` is one asking-after, for a program that wants to show progress
between them; `waitVideo` is the loop. Sora, Veo and xAI's video API are
behind the same three calls.

**There is no decoder here, for pictures or for video.** What the library
does with a file is send it, receive it, and name its type from the first
few bytes. Looking inside one is the program's business - the examples use
[fluxion-image](https://github.com/kisstp2006/fluxion-image) to open the
PNGs that come back and to draw the one they send - and a program that
depends on this library never fetches it.

## The key goes to the provider, and nowhere else

A picture's link points at somebody's storage, and a redirect can point
anywhere. The key, and the provider's extra headers, are sent to the
provider's own scheme, host and port only; a link or a redirect to anywhere
else is followed without them. The tests check this with a second server
that refuses any request carrying a key.

## A failure says what happened

A Zig error cannot carry a message, and the provider's message is usually
the part worth reading. Every call clears `client.failure` when it starts and
fills it when it fails:

```zig
var answer = client.chat(request) catch |err| {
    std.debug.print("{t}: {s}\n", .{ err, client.failure.message() });
    // error.OutOfCredit: Insufficient Balance
    return err;
};
```

The errors say which kind of failure it was, the same way for every
provider: `Unauthorized` is a wrong key whether it came as OpenAI's 401 or
Google's 400, `OutOfCredit` is an empty account whether it came as a 402 or as
OpenAI's 429 with `insufficient_quota`, and `RateLimited` is only ever worth
waiting out. The network's own errors come through as `std.http.Client`
returns them, with `failure` saying what was being attempted.

## What is not modelled is still reachable

The types cover what every provider shares. For the rest:

- **`extra`** on every request is the text of a JSON object whose members go
  into the body as they are, replacing a field of the same name:
  `.extra = "{\"reasoning_effort\":\"low\"}"`, DeepSeek's
  `{"thinking":{"type":"enabled"}}`, Gemini's `safetySettings`, tools.
- **`raw`** on every answer is the body as it came: tool calls, citations,
  log probabilities, to be read with `std.json`.
- **`Client.call`** sends anything to any endpoint with the provider's key,
  headers and error handling: embeddings, OpenAI's Responses API, a
  provider's own extras.

Tool calling, audio and embeddings have no types of their own yet; the
three above reach all of them.

## Install

```bash
zig fetch --save git+https://github.com/kisstp2006/fluxion-ai
```

Or, for a checkout next to your project, add to `build.zig.zon`:

```zig
.dependencies = .{
    .fluxion_ai = .{ .path = "../fluxion-ai" },
},
```

Either way, wire it up in `build.zig`:

```zig
const fluxion_ai = b.dependency("fluxion_ai", .{ .target = target, .optimize = optimize });
exe_mod.addImport("fluxion_ai", fluxion_ai.module("fluxion_ai"));
```

A program behind a proxy calls `client.http.initDefaultProxies(arena,
environ_map)` once, before the first request, and `HTTPS_PROXY` is honoured
from then on.

## Examples

```bash
zig build chat -- --provider deepseek "Why is the sky blue?"
zig build stream -- --provider anthropic "Write a haiku about Zig."
zig build models -- --provider gemini
zig build image -- --provider gemini "a lighthouse in a storm, oil on canvas"
zig build vision -- --provider openai
zig build video -- --seconds 4 "a paper boat drifting down a rainy street"
```

Each reads its key from the provider's usual variable and has a default
model for each provider it knows; `--model` picks another, and `--base-url`
points any of them at another OpenAI-compatible server. `image`, `vision`
and `video` use fluxion-image: to report what is inside the PNGs that come
back, to draw the test card `vision` asks about, and to read the size of a
first frame, which Sora wants the video to match. What they make goes in
`zig-out/`.

## Build

```bash
zig build test        # the unit tests, then the whole way down against local servers
zig build examples    # build every example into zig-out/bin
zig build docs        # generate API docs into zig-out/docs
```

The tests need no key and no network beyond this machine: two
`std.http.Server`s run beside them, one playing a provider in all three
shapes and one playing somebody's storage.

On Windows, Zig 0.16.0 reports a refused connection - Ollama not running,
say - as `error.Unexpected`, with a stack trace in debug builds; `failure`
still names the address that refused.

## Licence

`SPDX-License-Identifier: BSL-1.0`

[Boost Software License 1.0](LICENSE): use it, change it, ship it, in
anything. The copyright notice and the licence text travel with the source;
a binary built from it carries nothing. The examples' fluxion-image is
BSD-2-Clause, and is fetched for the examples only.
