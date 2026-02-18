#!/bin/bash
# --- المسار لملف السجل (Log) ---
LOG_FILE="/home/unitree/gemini_startup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- بدأ تشغيل سكربت Gemini في: $(date) ---"

# --- انتظار لمدة 5 ثوانٍ ---
echo "الانتظار لمدة 5 ثوانٍ لتهيئة أجهزة الصوت..."
sleep 15

# --- إعدادات الصوت (PulseAudio) ---
export PULSE_RUNTIME_PATH="/run/user/$(id -u)/pulse/"
echo "تطبيق إعدادات الصوت الافتراضية لـ Anker PowerConf..."
pactl load-module module-udev-detect
pactl set-default-sink alsa_output.usb-Anker_PowerConf_A3321-DEV-SN1-01.iec958-stereo
pactl set-default-source alsa_input.usb-Anker_PowerConf_A3321-DEV-SN1-01.mono-fallback
echo "تم تطبيق إعدادات الصوت."

# --- المسار إلى ملف إعدادات Conda ---
CONDA_SETUP_SCRIPT="/home/unitree/miniconda3/etc/profile.d/conda.sh"
PYTHON_SCRIPT="/home/unitree/gemini.py"

# 1. تحميل إعدادات Conda
echo "تحميل إعدادات Conda..."
source "$CONDA_SETUP_SCRIPT"

# 2. تفعيل بيئة 'base'
echo "تفعيل بيئة 'base'..."
source /home/unitree/miniconda3/bin/activate base

# 3. إضافة مفتاح API
export GEMINI_API_KEY="GEMINI_API_KEY"

# 4. حلقة لا نهائية لإعادة تشغيل السكربت
echo "=== بدء حلقة إعادة التشغيل التلقائي ==="
while true; do
    echo "--- بدء تشغيل gemini.py في: $(date) ---"
    python "$PYTHON_SCRIPT"
    EXIT_CODE=$?
    echo "--- انتهى سكربت البايثون في: $(date) بكود الخروج: $EXIT_CODE ---"
    echo "إعادة التشغيل بعد 5 ثوانٍ..."
    sleep 5
done
