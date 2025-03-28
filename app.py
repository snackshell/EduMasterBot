import os
import re
import logging
import urllib.parse
import time
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from openai import OpenAI, APITimeoutError, APIError
from telegram.constants import ParseMode
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in environment variables")

# Fix the API key loading
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_KEY not found in environment variables. Please set it in your .env file.")

MODEL = "google/gemini-2.5-pro-exp-03-25:free"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Initialize OpenAI client with explicit API key and proper headers
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    timeout=30.0,
    default_headers={
        "HTTP-Referer": "https://t.me/EduMasterBot",  # Optional but recommended
        "X-Title": "EduMasterBot",  # Optional but recommended
        "Authorization": f"Bearer {OPENROUTER_KEY}"  # Add explicit Authorization header
    }
)

# User context storage
user_contexts = {}

# Update the subjects lists to focus on Chemistry, English, and Biology
NATURAL_SUBJECTS = ["Chemistry", "Biology", "English"]
SOCIAL_SUBJECTS = ["Geography", "History", "Economics"]  # Keep these as alternatives

# Enhanced system prompt with updated subject focus
SYSTEM_PROMPT = """You are EduMaster, an expert tutor for Ethiopian Grade 12 exams.

IMPORTANT FORMATTING INSTRUCTIONS:
1. For bold text, use **double asterisks** (standard Markdown)
2. For italic text, use _underscores_ (Telegram format)
3. For mathematical formulas, use simple notation without special characters
4. Keep answers under 4000 characters total to fit in Telegram messages
5. Avoid using triple backticks - use single backticks for inline code

CONTENT RULES:
- Focus on Ethiopian Grade 12 curriculum for Chemistry, Biology, and English subjects
- Provide bilingual explanations (English/Amharic) when appropriate
- Always include concrete examples and complete explanations
- For science problems, show detailed step-by-step solutions
- Format your answers to be readable in Telegram
"""

# Function to process AI response for proper Telegram formatting
def process_ai_response(text):
    """Convert common markdown to Telegram-compatible formatting"""
    if not text:
        return "No response received"
    
    # Convert LaTeX formulas enclosed in $$ to code blocks
    text = re.sub(r'\$\$(.*?)\$\$', r'```\n\1\n```', text, flags=re.DOTALL)
    
    # Convert inline LaTeX formulas enclosed in $ to inline code
    text = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', r'`\1`', text)
    
    # Convert __italic__ to _italic_ (standardize)
    text = re.sub(r'__(.*?)__', r'_\1_', text)
    
    # Fix common Markdown issues that cause parsing errors
    # Replace triple backticks with single backticks for code blocks
    text = re.sub(r'```(?:math)?\n?(.*?)```', r'`\1`', text, flags=re.DOTALL)
    
    # Ensure asterisks are balanced
    open_count = text.count('**') // 2
    if open_count * 2 != text.count('**'):
        # Unbalanced asterisks, replace all with plain text
        text = text.replace('**', '')
    
    # Ensure underscores are balanced
    open_count = text.count('_') // 2
    if open_count * 2 != text.count('_'):
        # Unbalanced underscores, replace all with plain text
        text = text.replace('_', '')
    
    return text

def split_message(text, max_length=4000):
    """Split a message into parts that fit within Telegram's message size limits"""
    if not text:
        return ["No response available"]
    
    if len(text) <= max_length:
        return [text]
    
    parts = []
    
    # Try to split at paragraph breaks first
    paragraphs = text.split('\n\n')
    current_part = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit
        if len(current_part) + len(paragraph) + 2 > max_length:
            # If the current part is not empty, add it to parts
            if current_part:
                parts.append(current_part)
                current_part = ""
            
            # If the paragraph itself is too long, split it further
            if len(paragraph) > max_length:
                # Split at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_part) + len(sentence) + 1 > max_length:
                        if current_part:
                            parts.append(current_part)
                            current_part = sentence
                        else:
                            # If a single sentence is too long, split by character
                            sentence_parts = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
                            parts.extend(sentence_parts[:-1])
                            current_part = sentence_parts[-1]
                    else:
                        if current_part:
                            current_part += " " + sentence
                        else:
                            current_part = sentence
            else:
                # Start a new part with this paragraph
                current_part = paragraph
        else:
            # Add this paragraph to the current part
            if current_part:
                current_part += "\n\n" + paragraph
            else:
                current_part = paragraph
    
    # Add the last part if it's not empty
    if current_part:
        parts.append(current_part)
    
    return parts

# Enhanced keyboard functions
def stream_keyboard():
    return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Natural Science 🌿", callback_data="natural"),
                InlineKeyboardButton("Social Science 📚", callback_data="social"),
            ],
            [
            InlineKeyboardButton("Channel", url="https://t.me/Tigrai_Academy"),
            InlineKeyboardButton("Group", url="https://t.me/Tigrai_Academy"),
        ],
    ])

def remedial_keyboard():
    return InlineKeyboardMarkup([
            [InlineKeyboardButton("Regular Student", callback_data="regular")],
            [InlineKeyboardButton("Remedial Student", callback_data="remedial")],
    ])

def subject_keyboard(stream):
    subjects = NATURAL_SUBJECTS if stream == "natural" else SOCIAL_SUBJECTS
    keyboard = []
    row = []
    
    for i, subject in enumerate(subjects):
        if i > 0 and i % 2 == 0:
            keyboard.append(row)
            row = []
        row.append(InlineKeyboardButton(subject, callback_data=f"subject_{subject.lower().replace(' ', '_')}"))
    
    if row:
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("Back to Stream Selection", callback_data="back_to_stream")])
    
    return InlineKeyboardMarkup(keyboard)

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_contexts[user_id] = {
        "stream": None, 
        "remedial": False, 
        "subject": None,
        "history": []  # Add chat history storage
    }

    welcome_text = "📚 **Welcome to EduGeniusBot**\n\nPlease select your stream:"

    try:
        # Create keyboard first to avoid delays
        keyboard = stream_keyboard()
        
        # Send message with increased timeout
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_text,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30,
            pool_timeout=30,
        )
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        try:
            await update.message.reply_text("⚠️ Error starting the bot. Please try again.")
        except Exception as e2:
            logger.error(f"Critical error in start command fallback: {e2}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "**EduMasterBot Help**\n\n"
        "/start - Start the bot\n"
        "/help - Display this help message\n"
        "/subject - Select a specific subject\n"
        "/status - Check bot service status\n"
        "/clear - Clear your chat history\n\n"
        "**Group Chat Usage:**\n"
        "- Mention the bot: @EduMasterBot your question\n"
        "- Reply to the bot's message with your follow-up\n\n"
        "**Subjects:**\n"
        "- Chemistry\n"
        "- Biology\n"
        "- English\n\n"
        "[Tigray Academy Channel](https://t.me/Tigrai_Academy)"
    )
    
    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=help_text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        await send_safe_message(update, "⚠️ Error displaying help message. Please try again.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check all services
    try:
        # Check Telegram API
        telegram_status = "✅ Operational"
        
        # Check OpenRouter API with a simple request
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a test assistant"},
                    {"role": "user", "content": "Respond with 'OK' only"},
                ],
                max_tokens=5,
            )
            response_time = time.time() - start_time
            ai_status = f"✅ Operational ({response_time:.2f}s)"
        except Exception as e:
            logger.error(f"AI service error: {str(e)}")
            ai_status = "❌ Not responding"
        
        status_text = (
            "*EduGeniusBot Status*\n\n"
            f"*Telegram API:* {telegram_status}\n"
            f"*AI Service:* {ai_status}\n\n"
            f"*Bot Version:* 1.2.1\n"
            f"*Model:* {MODEL}"
        )
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=status_text,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await send_safe_message(update, "⚠️ Error checking status. Please try again.")

async def subject_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_contexts or not user_contexts[user_id].get("stream"):
        await send_safe_message(update, "Please use /start first to select your stream.")
        return
    
    stream = user_contexts[user_id]["stream"]
    
    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Select a subject for {stream.capitalize()} stream:",
            reply_markup=subject_keyboard(stream),
        )
    except Exception as e:
        logger.error(f"Error in subject command: {e}")
        await send_safe_message(update, "⚠️ Error displaying subjects. Please try again.")

# Callback handlers
async def set_stream(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    stream = query.data

    user_contexts[user_id]["stream"] = stream
    
    try:
        await query.edit_message_text(
            text=f"Selected: *{stream.capitalize()} Stream*\nAre you a Regular or Remedial student?",
            reply_markup=remedial_keyboard(),
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as e:
        logger.error(f"Error in set_stream: {e}")
        await send_callback_error(update, "Error setting stream. Please try again.")

async def set_remedial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if user_id not in user_contexts:
        await query.edit_message_text(
            text="Session expired. Please use /start to begin again."
        )
        return
    
    remedial_status = query.data == "remedial"
    user_contexts[user_id]["remedial"] = remedial_status
    
    stream = user_contexts[user_id]["stream"]
    
    try:
        # Using plain text without markdown to avoid escaping issues
        text = f"Configuration Complete\n\n• Stream: {stream.capitalize()}\n• Mode: {'Remedial' if remedial_status else 'Regular'}\n\nYou can now ask questions or use /subject to select a specific subject."
        
        await query.edit_message_text(
            text=text,
            # No parse mode to avoid escaping issues
        )
    except Exception as e:
        logger.error(f"Error in set_remedial: {e}")
        await send_callback_error(update, "Error setting student type. Please try again.")

async def set_subject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "back_to_stream":
        await query.edit_message_text(
            text="Please select your stream:",
            reply_markup=stream_keyboard(),
        )
        return
    
    if user_id not in user_contexts:
        await query.edit_message_text(
            text="Session expired. Please use /start to begin again."
        )
        return
    
    subject = query.data.replace("subject_", "").replace("_", " ").title()
    user_contexts[user_id]["subject"] = subject
    
    try:
        # Using plain text without markdown to avoid escaping issues
        await query.edit_message_text(
            text=f"Subject set to {subject}\n\nYou can now ask questions specific to this subject."
        )
    except Exception as e:
        logger.error(f"Error in set_subject: {e}")
        await send_callback_error(update, "Error setting subject. Please try again.")

# Helper functions for reliable message sending
async def send_safe_message(update, text, parse_mode=None):
    """Send a message with fallback to ensure delivery"""
    try:
        await update.message.reply_text(text, parse_mode=parse_mode)
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        try:
            # Try without parse mode as fallback
            await update.message.reply_text(text)
        except Exception as e2:
            logger.error(f"Critical error sending message: {e2}")

async def send_callback_error(update, text):
    """Send error message for callback queries"""
    query = update.callback_query
    try:
        # Try to answer callback if not already answered
        await query.answer(text=text[:200])  # Telegram limits callback answers to 200 chars
    except Exception:
        pass
    
    try:
        # Try to send a new message instead of editing
        await update.effective_chat.send_message(text)
    except Exception as e:
        logger.error(f"Critical error handling callback error: {e}")

# Main question handling with retry logic and fallback models
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if this is a group chat
    is_group = update.effective_chat.type in ["group", "supergroup"]
    
    # For group chats, only respond if the bot is mentioned or replied to
    if is_group:
        message = update.message
        bot_username = context.bot.username
        
        # Check if bot was mentioned or replied to
        was_mentioned = False
        if message.text and f"@{bot_username}" in message.text:
            # Bot was mentioned, remove the mention from the question
            question = message.text.replace(f"@{bot_username}", "").strip()
            was_mentioned = True
        elif message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id:
            # Message is a reply to the bot
            question = message.text
            was_mentioned = True
        
        if not was_mentioned:
            # Bot wasn't mentioned or replied to, don't respond
            return
    else:
        # Direct message to the bot
        question = update.message.text
    
    user_id = update.effective_user.id
    if user_id not in user_contexts:
        # Initialize user context for new users
        user_contexts[user_id] = {
            "stream": "natural",  # Default to natural science
            "remedial": False,
            "subject": "chemistry",  # Default to chemistry
            "history": []
        }
        
        # In group chats, don't send the welcome message
        if not is_group:
            await update.message.reply_text(
                "Welcome! I've set your default subject to Chemistry. "
                "Use /subject to change it or /start to set up your preferences."
            )
    
    user_data = user_contexts[user_id]
    
    # Prepare context-aware prompt
    stream_text = f"Stream: {user_data.get('stream', 'Natural Science')}"
    subject_text = f"Subject: {user_data.get('subject', 'Chemistry')}"
    remedial_text = "Type: Remedial student" if user_data.get('remedial') else "Type: Regular student"
    
    # Initialize history if not present
    if "history" not in user_data:
        user_data["history"] = []
    
    # Send typing action to indicate the bot is processing
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    except Exception as e:
        logger.warning(f"Could not send typing action: {e}")
    
    # Try different models in order of preference
    models = [MODEL, "anthropic/claude-3-haiku:free", "google/gemini-2.0-flash-exp:free"]
    response_received = False
    
    for model in models:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Trying model {model}, attempt {attempt+1}")
                # Send typing action again to keep it visible
                try:
                    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                except:
                    pass
                
                # Build messages array with system prompt, context info, and chat history
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                
                # Add context information as a system message
                context_info = f"{stream_text}\n{subject_text}\n{remedial_text}"
                messages.append({"role": "system", "content": f"User context: {context_info}"})
                
                # Add chat history (up to 4 previous exchanges)
                for msg in user_data["history"][-4:]:  # Limit to last 4 messages
                    messages.append(msg)
                
                # Add current question
                messages.append({"role": "user", "content": question})
                
                # Update the API request parameters to limit token count
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=30,
                    max_tokens=1000,  # Reduced from 2000 to 1000 for Telegram compatibility
                    temperature=0.7,  # Add temperature for more controlled responses
                    extra_headers={
                        "HTTP-Referer": "https://t.me/EduMasterBot",
                        "X-Title": "EduMasterBot"
                    }
                )
                
                # Improve error handling for API responses
                try:
                    raw_answer = completion.choices[0].message.content
                    if not raw_answer:
                        raise ValueError("Empty response from API")
                except (AttributeError, IndexError, ValueError) as e:
                    logger.error(f"Invalid response format from {model}: {str(e)}")
                    continue  # Try next attempt or model
                
                # Process the response for proper Telegram formatting
                answer = process_ai_response(raw_answer)
                
                # Store the exchange in history
                user_data["history"].append({"role": "user", "content": question})
                user_data["history"].append({"role": "assistant", "content": raw_answer})
                
                # Keep history to a reasonable size (max 10 messages)
                if len(user_data["history"]) > 10:
                    user_data["history"] = user_data["history"][-10:]
                
                # For group chats, include the question in the response
                if is_group:
                    # Get the user's first name
                    user_name = update.effective_user.first_name
                    answer = f"**Question from {user_name}:**\n{question}\n\n**Answer:**\n{answer}"
                
                # Split long messages
                message_parts = split_message(answer)
                for part in message_parts:
                    try:
                        # Send with Telegram's Markdown parsing
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=part,
                            parse_mode=ParseMode.MARKDOWN,
                            reply_to_message_id=update.message.message_id if is_group else None
                        )
                    except Exception as msg_error:
                        # If markdown fails, try to fix common issues
                        logger.warning(f"Markdown rendering failed: {msg_error}")
                        
                        # Remove all markdown formatting as a last resort
                        clean_text = re.sub(r'[*_`]', '', part)
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=clean_text,
                            reply_to_message_id=update.message.message_id if is_group else None
                        )
                        
                response_received = True
                break  # Success, exit the retry loop
                
            except APITimeoutError:
                logger.warning(f"API timeout with model {model} on attempt {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    
            except APIError as e:
                logger.error(f"API error with model {model}: {str(e)}")
                break  # Try next model
                
            except Exception as e:
                logger.error(f"Error processing question with model {model}: {str(e)}")
                break  # Try next model
        
        if response_received:
            break  # If we got a response, don't try other models
    
    # If all models and attempts failed
    if not response_received:
        await update.message.reply_text(
            "⚠️ I'm having trouble connecting to the AI service right now. Please try again in a few moments.",
            reply_to_message_id=update.message.message_id if is_group else None
        )

def safe_markdown_v2(text):
    """Safely escape text for Markdown V2 format"""
    if not text:
        return ""
    escape_chars = r'_*[]()~`>#+-=|{}.!\\'
    return re.sub(r'([' + re.escape(escape_chars) + r'])', r'\\\1', text)

# Add a new command to clear chat history
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_contexts and "history" in user_contexts[user_id]:
        user_contexts[user_id]["history"] = []
        await update.message.reply_text("✅ Your chat history has been cleared.")
    else:
        await update.message.reply_text("No chat history to clear.")

def main():
    # Create the Application with increased timeouts
    app = ApplicationBuilder().token(TOKEN).get_updates_read_timeout(42).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("subject", subject_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("clear", clear_history))  # Add new command
    
    # Register callback handlers
    app.add_handler(CallbackQueryHandler(set_stream, pattern="^(natural|social)$"))
    app.add_handler(CallbackQueryHandler(set_remedial, pattern="^(regular|remedial)$"))
    app.add_handler(CallbackQueryHandler(set_subject, pattern="^(subject_|back_to_stream)"))
    
    # Register message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Start the Bot with increased timeouts
    app.run_polling(timeout=60, poll_interval=1.0)

if __name__ == "__main__":
    main()
