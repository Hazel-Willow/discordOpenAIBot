import asyncio
import os
import logging
import discord
import httpx
import yaml
import time
import re
import json
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from discord.app_commands import Choice
from discord.ext import commands
from openai import AsyncOpenAI
from urllib.parse import urlparse
from openai import OpenAI


## Declaration of Variables
ALLOWED_SPECIALS = '<>@!,.;-:"()*^/#_{}[]\\&%$|'

_SANITIZE_RE = re.compile(rf'[^A-Za-z0-9\s{re.escape(ALLOWED_SPECIALS)}]')

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "mistralai", "magistral-small-2509", "qwen_qwen3-vl-30b-a3b-thinking", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai", "llama-3.2-3b-instruct")

MAX_MESSAGE_NODES = 500

## Logging formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

def sanitize_text(sanitary: str | None) -> str:
    if not sanitary:
        return ""
    return _SANITIZE_RE.sub("", sanitary)

## Get config and verify syntax
def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)
    
config = get_config()
curr_model = next(iter(config["models"]))
bot_nickname = config["nickname"]

## Declaration of Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True 
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

## This allows the bot to pull images
httpx_client = httpx.AsyncClient()

## This is the metadata that the MsgNode tracks for a message
@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

## Declairing variables used by msg_nodes
msg_nodes = {}
last_task_time = 0
last_bot_msg_by_channel: dict[int, tuple[int, float]] = {}

def _update_bot_anchor(msg: discord.Message) -> None:
    ts = discord.utils.utcnow().timestamp()
    last_bot_msg_by_channel[msg.channel.id] = (msg.id, ts)

## Slash command for changing which model is actively being used
@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."
    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))

## When bot comes up, it displays an invite link with requested join permissions
## Permissions are set for messaging, attaching files, reacting to messages, and VC use
## VC, reacts, and uploading files are planned features, but not currently in use 
@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=414501411904&scope=bot\n")

    await discord_bot.tree.sync()

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    ## Bot replies to mentions instantly, waits 10 seconds before replying to bots, and
    ## waits 5 seconds before replying to bot_nickname
    if new_msg.author.bot:
        await asyncio.sleep(10)
        logging.info(f"OwO BOTS!: \n{new_msg.content}")
    if (not is_dm and bot_nickname in new_msg.content):
        await asyncio.sleep(5)
        logging.info(f"OwO!! Is this for me: \n{new_msg.content}")
    if (not is_dm and discord_bot.user not in new_msg.mentions and bot_nickname not in new_msg.content):
        return
    
    
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    ## This is the permissions logic. It's for allowing/blocking users and channels
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return
    
    ##LLM definitions/options
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)
    extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    ## Build message chain and set user warnings
    messages = []
    curr_msg = new_msg

    ## Message handling
    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                ## Removes userID and then sanatizes content
                raw_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                cleaned_content = sanitize_text(raw_content)

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                ## Sanatizes embeds
                def _embed_texts(e: discord.Embed) -> list[str]:
                    t = sanitize_text(e.title)
                    d = sanitize_text(e.description)
                    f = sanitize_text(getattr(getattr(e, "footer", None), "text", None))
                    return [x for x in (t, d, f) if x]

                ## Sanatizes text files, fr fr
                text_attachment_bodies = [
                    sanitize_text(resp.text)
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("text")
                ]

                ## Puts shit back together
                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(_embed_texts(embed)) for embed in curr_msg.embeds]
                    + text_attachment_bodies
                )

                ## Reattaches images
                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                ##-------++++++++-------++++++++-------++++++++-------++++++++-------##
                ##                    RESUME REWRITING CODE HERE                     ##
                ##-------++++++++-------++++++++-------++++++++-------++++++++-------##

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    ## Checks if message is a reply. If the message replied to is deleted, or there is no message 
                    ## being replied to, the logic continues to the next block after setting the current message as the parent
                    ref = curr_msg.reference
                    if ref and getattr(ref, "message_id", None):
                        parent = getattr(ref, "cached_message", None)
                        if parent is None:
                            parent = await curr_msg.channel.fetch_message(ref.message_id)
                        curr_node.parent_msg = parent

                    else:
                        ## Fetches current message channel
                        ch = curr_msg.channel

                        ## Defines group chats and DMS as "is_dm_like"
                        is_dm_like = ch.type in (
                            discord.ChannelType.private,
                            discord.ChannelType.group
                        )

                        ## Defines public and private thread variables and sets the channel type as is_public_thread or is_private_thread if true
                        is_public_thread = ch.type == discord.ChannelType.public_thread
                        is_private_thread = ch.type == discord.ChannelType.private_thread

                        ## Checks if bot was mentioned instead
                        bot_mentioned = discord_bot.user in getattr(curr_msg, "mentions", [])
                        
                        ## Checks to see if the message is in a thread, retrieves entire thread, catches exception if unable to retrieve thread
                        starter_msg = None
                        if is_public_thread or is_private_thread:
                            starter_msg = ch.starter_message
                            if starter_msg is None and ch.parent and ch.parent.type == discord.ChannelType.text:
                                try:
                                    starter_msg = await ch.parent.fetch_message(ch.id)
                                except (discord.NotFound, discord.HTTPException):
                                    starter_msg = None

                        ## Currently magic
                        ## Define me plz
                        session_parent = None
                        session_info = last_bot_msg_by_channel.get(ch.id)
                        if session_info:
                            session_parent_id, session_parent_ts = session_info
                            if (discord.utils.utcnow().timestamp() - session_parent_ts) <= 300:
                                try:
                                    session_parent = await ch.fetch_message(session_parent_id)
                                except (discord.NotFound, discord.HTTPException):
                                    session_parent = None

                        ## If DMs or group chat, previous message is defined as chat history
                        if is_dm_like:
                            prev_msg = ([m async for m in ch.history(before=curr_msg, limit=10)] or [None])[0]
                            if prev_msg and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                                curr_node.parent_msg = prev_msg
                            else:
                                curr_node.parent_msg = None

                        ## If threads, pulls history of the thread
                        elif is_public_thread or is_private_thread:
                            if bot_mentioned:
                                curr_node.parent_msg = session_parent or starter_msg
                            else:
                                if starter_msg and starter_msg.author == discord_bot.user:
                                    curr_node.parent_msg = session_parent or starter_msg
                                else:
                                    curr_node.parent_msg = session_parent

                        ## If bot mentioned, pulls reply chain
                        ## If bot nickname used, pulls chat log up to message history limit
                        else:
                            if bot_mentioned:
                                curr_node.parent_msg = session_parent
                            else:
                                prev_msg = ([m async for m in ch.history(before=curr_msg, limit=10)] or [None])[0]
                                if prev_msg and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                                    curr_node.parent_msg = prev_msg
                                else:
                                    curr_node.parent_msg = None
                
                ## Throws exception if no message is able to be pulled, and outputs to the terminal
                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            ## Checks if reply chain contains the maximum images or text
            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            ## Checks if the message is empty
            ## Adds the username of the message if the author isn't defined, then adds it to the beginning of the message if it is
            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    ## Adds system prompt if present, adds system prompt if present and informs system of Discord IDs
    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        system_prompt += "\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        messages.append(dict(role="system", content=system_prompt))

    ## Generate and send response message
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    max_message_length = 2000

    ## Collect chunks from LLM, combine them, and send them
    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    try:
        ## Show bot is typing while response is being generated
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

            for content in response_contents:
                ## Replies to message 
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                response_msgs.append(response_msg)

                _update_bot_anchor(response_msg)

                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass

