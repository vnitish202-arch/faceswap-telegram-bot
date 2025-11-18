#!/usr/bin/env python3

import os
import io
import time
import sqlite3
import logging
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import face_recognition

from telegram import Update, Bot, InputMediaPhoto
from telegram.ext import (
    Updater, CommandHandler, MessageHandler,
    Filters, CallbackContext
)

# ==============================
# SAFE CONFIGURATION
# ==============================

# DO NOT hardcode your token here.
# Put your token in GitHub Secrets → BOT_TOKEN
BOT_TOKEN = os.environ.get("8183535358:AAFJqa1yNC3EqwlwcTPHArjH_GzIrpSa2Qw")

# Admin ID can stay in code (not sensitive)
ADMIN_CHAT_ID = 7888759188

BOT_NAME = "ConsensualSwapBot"
WORKDIR = "jobs"
DB_PATH = "jobs.db"
LOGFILE = "bot.log"

os.makedirs(WORKDIR, exist_ok=True)

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("swapbot")

# ==============================
# DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            chat_id INTEGER,
            target_path TEXT,
            source_path TEXT,
            consent_text TEXT,
            status TEXT,
            created_at TEXT,
            processed_at TEXT,
            output_path TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def create_job(uid, cid, tpath, spath, consent):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO jobs (user_id, chat_id, target_path, source_path, consent_text, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending_review', ?)
    """, (uid, cid, tpath, spath, consent, now))
    jid = cur.lastrowid
    conn.commit()
    conn.close()
    return jid

# ==============================
# IMAGE PROCESSING
# ==============================
def pil_to_cv2(pimg): return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
def cv2_to_pil(cimg): return Image.fromarray(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))

def watermark(img, text):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    msg = f"{text} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    draw.text((10, h - 25), msg, fill="white")
    return img

def detect_landmarks(rgb):
    loc = face_recognition.face_locations(rgb)
    if not loc: return []
    return face_recognition.face_landmarks(rgb, loc)

def extract_mask(bgr, lm):
    pts = []
    for key in ['chin','left_eyebrow','right_eyebrow','nose_tip','top_lip','bottom_lip']:
        pts += lm.get(key, [])
    if not pts: return None
    hull = cv2.convexHull(np.array(pts))
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def swap_face(tpath, spath, outpath):
    t_pil = Image.open(tpath).convert("RGB")
    s_pil = Image.open(spath).convert("RGB")

    t = pil_to_cv2(t_pil)
    s = pil_to_cv2(s_pil)

    t_lm = detect_landmarks(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
    s_lm = detect_landmarks(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))

    if not t_lm or not s_lm:
        raise ValueError("Face not detected")

    t_lm = t_lm[0]
    s_lm = s_lm[0]

    s_mask = extract_mask(s, s_lm)
    if s_mask is None:
        raise ValueError("Mask fail")

    ys, xs = np.where(s_mask == 255)
    h_s = ys.max() - ys.min()
    w_s = xs.max() - xs.min()

    t_mask = extract_mask(t, t_lm)
    yt, xt = np.where(t_mask == 255)
    h_t = yt.max() - yt.min()
    w_t = xt.max() - xt.min()

    scale = min(w_t / (w_s + 1), h_t / (h_s + 1))
    scale = max(0.4, min(scale, 4.0))

    new_w = int(s.shape[1] * scale)
    new_h = int(s.shape[0] * scale)

    s_res = cv2.resize(s, (new_w, new_h))
    m_res = cv2.resize(s_mask, (new_w, new_h))

    nose = t_lm['nose_tip']
    center = tuple(np.mean(nose, axis=0).astype(int))

    result = cv2.seamlessClone(s_res, t, m_res, center, cv2.NORMAL_CLONE)
    final = cv2_to_pil(result)
    final = watermark(final, BOT_NAME)
    final.save(outpath)

    return outpath

# ==============================
# BOT HANDLERS
# ==============================
TEMP = {}
CONSENT_WORD = "I CONSENT - I OWN THE RIGHTS OR HAVE PERMISSION"

updater = Updater(BOT_TOKEN, use_context=True)
dispatcher = updater.dispatcher
bot = updater.bot

def start(update, ctx):
    update.message.reply_text(
        "Welcome! Send /swap to begin.\n"
        "Upload TARGET → SOURCE → then send consent phrase:\n\n"
        f"`{CONSENT_WORD}`",
        parse_mode="Markdown"
    )

def swap(update, ctx):
    uid = update.message.from_user.id
    TEMP[uid] = {"t": None, "s": None}
    update.message.reply_text("Send TARGET image")

def photo(update, ctx):
    uid = update.message.from_user.id
    if uid not in TEMP:
        update.message.reply_text("Send /swap first")
        return

    file = update.message.photo[-1].get_file()
    buf = io.BytesIO()
    file.download(out=buf)
    buf.seek(0)

    if TEMP[uid]["t"] is None:
        path = f"jobs/target_{uid}_{int(time.time())}.jpg"
        with open(path, "wb") as f: f.write(buf.read())
        TEMP[uid]["t"] = path
        update.message.reply_text("TARGET saved. Now send SOURCE.")
        return

    if TEMP[uid]["s"] is None:
        path = f"jobs/source_{uid}_{int(time.time())}.jpg"
        with open(path, "wb") as f: f.write(buf.read())
        TEMP[uid]["s"] = path
        update.message.reply_text(f"SOURCE saved.\nNow send:\n`{CONSENT_WORD}`", parse_mode="Markdown")

def text(update, ctx):
    uid = update.message.from_user.id
    msg = update.message.text.strip()

    if uid not in TEMP:
        return update.message.reply_text("Send /swap first")

    if TEMP[uid]["t"] and TEMP[uid]["s"] and msg == CONSENT_WORD:
        jid = create_job(uid, update.message.chat_id, TEMP[uid]["t"], TEMP[uid]["s"], msg)
        TEMP.pop(uid)
        update.message.reply_text(f"Job #{jid} submitted.")
        bot.send_message(ADMIN_CHAT_ID, f"New job #{jid}. Use /review {jid}")
        return

    update.message.reply_text("Consent phrase incorrect.")

# ==============================
# ADMIN COMMANDS
# ==============================
def review(update, ctx):
    if update.message.from_user.id != ADMIN_CHAT_ID:
        return

    args = ctx.args
    if not args:
        return update.message.reply_text("Usage: /review <id>")

    jid = int(args[0])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT target_path, source_path FROM jobs WHERE id=?", (jid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return update.message.reply_text("Job not found")

    t, s = row

    bot.send_media_group(
        ADMIN_CHAT_ID,
        [
            InputMediaPhoto(open(t, "rb"), caption="TARGET"),
            InputMediaPhoto(open(s, "rb"), caption="SOURCE")
        ]
    )
    update.message.reply_text(f"/approve {jid} or /reject {jid}")

def approve(update, ctx):
    if update.message.from_user.id != ADMIN_CHAT_ID:
        return

    jid = int(ctx.args[0])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chat_id, target_path, source_path FROM jobs WHERE id=?", (jid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return update.message.reply_text("Not found")

    chat_id, t, s = row
    out = f"jobs/out_{jid}.jpg"

    try:
        swap_face(t, s, out)
        bot.send_photo(chat_id, open(out, "rb"), caption=f"Job #{jid} done")
        update.message.reply_text("Approved + processed")
    except Exception as e:
        update.message.reply_text(f"Error: {e}")

def reject(update, ctx):
    if update.message.from_user.id != ADMIN_CHAT_ID:
        return

    jid = int(ctx.args[0])
    update.message.reply_text(f"Job {jid} rejected")

# ==============================
# REGISTER HANDLERS
# ==============================
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("swap", swap))
dispatcher.add_handler(CommandHandler("review", review))
dispatcher.add_handler(CommandHandler("approve", approve))
dispatcher.add_handler(CommandHandler("reject", reject))

dispatcher.add_handler(MessageHandler(Filters.photo, photo))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, text))

# ==============================
# START BOT
# ==============================
if __name__ == "__main__":
    print("Bot running...")
    updater.start_polling()
    updater.idle()
