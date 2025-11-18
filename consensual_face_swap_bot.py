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

# Load token from env (recommended). If you insist on hardcoding, replace below.
BOT_TOKEN = os.environ.get("8183535358:AAFJqa1yNC3EqwlwcTPHArjH_GzIrpSa2Qw")
if not BOT_TOKEN:
    print("ERROR: BOT_TOKEN environment variable not set. Exiting.")
    raise SystemExit(1)

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
def pil_to_cv2(pimg):
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cimg):
    return Image.fromarray(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))

def watermark(img, text):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    msg = f"{text} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    # try to use a truetype font if available, else default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textsize(msg, font=font)
    draw.rectangle((8, h - th - 12, 10 + tw + 4, h - 6), fill=(0, 0, 0, 180))
    draw.text((10, h - th - 10), msg, fill="white", font=font)
    return img

def detect_landmarks(rgb):
    loc = face_recognition.face_locations(rgb)
    if not loc:
        return []
    return face_recognition.face_landmarks(rgb, loc)

def extract_mask(bgr, lm):
    pts = []
    for key in ['chin','left_eyebrow','right_eyebrow','nose_tip','top_lip','bottom_lip']:
        pts += lm.get(key, [])
    if not pts:
        return None
    hull = cv2.convexHull(np.array(pts))
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def swap_face(tpath, spath, outpath):
    # Load images
    t_pil = Image.open(tpath).convert("RGB")
    s_pil = Image.open(spath).convert("RGB")

    t = pil_to_cv2(t_pil)
    s = pil_to_cv2(s_pil)

    # Detect landmarks
    t_lm_list = detect_landmarks(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
    s_lm_list = detect_landmarks(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))

    if not t_lm_list or not s_lm_list:
        raise ValueError("Face not detected in one of the images")

    t_lm = t_lm_list[0]
    s_lm = s_lm_list[0]

    s_mask = extract_mask(s, s_lm)
    if s_mask is None:
        raise ValueError("Mask creation failed for source")

    # compute source face bbox
    ys, xs = np.where(s_mask == 255)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Invalid mask area for source")
    h_s = ys.max() - ys.min()
    w_s = xs.max() - xs.min()

    # compute target face bbox
    t_mask = extract_mask(t, t_lm)
    yt, xt = np.where(t_mask == 255)
    if yt.size == 0 or xt.size == 0:
        raise ValueError("Invalid mask area for target")
    h_t = yt.max() - yt.min()
    w_t = xt.max() - xt.min()

    # scale source to target
    scale = min(w_t / (w_s + 1), h_t / (h_s + 1))
    scale = max(0.4, min(scale, 4.0))

    new_w = max(10, int(s.shape[1] * scale))
    new_h = max(10, int(s.shape[0] * scale))

    s_res = cv2.resize(s, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    m_res = cv2.resize(s_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # pick center from target nose_tip if available otherwise image center
    if 'nose_tip' in t_lm and t_lm['nose_tip']:
        nose = np.array(t_lm['nose_tip'])
        center = tuple(nose.mean(axis=0).astype(int))
    else:
        center = (t.shape[1]//2, t.shape[0]//2)

    try:
        result = cv2.seamlessClone(s_res, t, m_res, center, cv2.NORMAL_CLONE)
    except Exception as e:
        # fallback: simple paste
        out = t.copy()
        y0 = max(0, center[1] - new_h//2)
        x0 = max(0, center[0] - new_w//2)
        y1 = min(out.shape[0], y0 + new_h)
        x1 = min(out.shape[1], x0 + new_w)
        roi = out[y0:y1, x0:x1]
        mcut = m_res[0:(y1-y0), 0:(x1-x0)].astype(bool)
        roi[mcut] = s_res[0:(y1-y0), 0:(x1-x0)][mcut]
        out[y0:y1, x0:x1] = roi
        result = out

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

def start(update: Update, ctx: CallbackContext):
    update.message.reply_text(
        "Welcome! Send /swap to begin.\n"
        "Upload TARGET → SOURCE → then send consent phrase:\n\n"
        f"`{CONSENT_WORD}`",
        parse_mode="Markdown"
    )

def swap(update: Update, ctx: CallbackContext):
    uid = update.message.from_user.id
    TEMP[uid] = {"t": None, "s": None}
    update.message.reply_text("Send TARGET image")

def photo(update: Update, ctx: CallbackContext):
    uid = update.message.from_user.id
    if uid not in TEMP:
        update.message.reply_text("Send /swap first")
        return

    file = update.message.photo[-1].get_file()
    buf = io.BytesIO()
    file.download(out=buf)
    buf.seek(0)

    if TEMP[uid]["t"] is None:
        path = os.path.join(WORKDIR, f"target_{uid}_{int(time.time())}.jpg")
        with open(path, "wb") as f:
            f.write(buf.read())
        TEMP[uid]["t"] = path
        update.message.reply_text("TARGET saved. Now send SOURCE.")
        return

    if TEMP[uid]["s"] is None:
        path = os.path.join(WORKDIR, f"source_{uid}_{int(time.time())}.jpg")
        with open(path, "wb") as f:
            f.write(buf.read())
        TEMP[uid]["s"] = path
        update.message.reply_text(f"SOURCE saved.\nNow send:\n`{CONSENT_WORD}`", parse_mode="Markdown")
        return

def text(update: Update, ctx: CallbackContext):
    uid = update.message.from_user.id
    msg = update.message.text.strip()

    if uid not in TEMP:
        return update.message.reply_text("Send /swap first")

    if TEMP[uid]["t"] and TEMP[uid]["s"] and msg == CONSENT_WORD:
        jid = create_job(uid, update.message.chat_id, TEMP[uid]["t"], TEMP[uid]["s"], msg)
        TEMP.pop(uid, None)
        update.message.reply_text(f"Job #{jid} submitted.")
        bot.send_message(ADMIN_CHAT_ID, f"New job #{jid}. Use /review {jid}")
        return

    update.message.reply_text("Consent phrase incorrect.")

# ==============================
# ADMIN COMMANDS
# ==============================
def review(update: Update, ctx: CallbackContext):
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

    # send both images as media group (safe open with context manager)
    with open(t, "rb") as tf, open(s, "rb") as sf:
        bot.send_media_group(
            ADMIN_CHAT_ID,
            [
                InputMediaPhoto(tf, caption=f"TARGET #{jid}"),
                InputMediaPhoto(sf, caption=f"SOURCE #{jid}")
            ]
        )
    update.message.reply_text(f"/approve {jid} or /reject {jid}")

def approve(update: Update, ctx: CallbackContext):
    if update.message.from_user.id != ADMIN_CHAT_ID:
        return

    if not ctx.args:
        return update.message.reply_text("Usage: /approve <job_id>")

    jid = int(ctx.args[0])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chat_id, target_path, source_path FROM jobs WHERE id=?", (jid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return update.message.reply_text("Not found")

    chat_id, t, s = row
    out = os.path.join(WORKDIR, f"out_{jid}.jpg")

    try:
        swap_face(t, s, out)
        with open(out, "rb") as outf:
            bot.send_photo(chat_id, outf, caption=f"Job #{jid} done")
        # update DB status
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE jobs SET status='completed', processed_at=?, output_path=? WHERE id=?", (datetime.utcnow().isoformat(), out, jid))
        conn.commit()
        conn.close()
        update.message.reply_text("Approved + processed")
    except Exception as e:
        logger.exception("Processing error for job %s: %s", jid, e)
        update.message.reply_text(f"Error: {e}")

def reject(update: Update, ctx: CallbackContext):
    if update.message.from_user.id != ADMIN_CHAT_ID:
        return

    if not ctx.args:
        return update.message.reply_text("Usage: /reject <job_id>")

    jid = int(ctx.args[0])
    # set status in DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE jobs SET status='rejected', processed_at=? WHERE id=?", (datetime.utcnow().isoformat(), jid))
    conn.commit()
    conn.close()

    # notify user if chat_id exists
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chat_id FROM jobs WHERE id=?", (jid,))
    row = cur.fetchone()
    conn.close()
    if row:
        bot.send_message(row[0], f"Your job #{jid} was rejected by admin.")
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
    
