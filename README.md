```markdown
# EduMasterBot 🇪🇹
_A Telegram AI tutor for Ethiopian Grade 12 exam preparation_

## 📚 Features
- **Curriculum-Aligned** answers for:
  - Natural Science: Maths, Physics, Chemistry, Biology
  - Social Science: Maths (SS), Geography, History
- **Remedial & Regular** student modes
- **Formatted Responses**:
  ```math
  F = ma  # Mathematical formulas
  ```
  **Key Concepts** in bold  
  • Step-by-step solutions  
  • Bilingual explanations (English/Amharic)
- **PDF-Ready** content for all subjects

## 🚀 Quick Start
1. Get API keys:
   - [Telegram Bot Token](https://t.me/BotFather)
   - [OpenRouter API Key](https://openrouter.ai/)

2. Create `.env` file:
   ```env
   TELEGRAM_TOKEN=your_bot_token
   OPENROUTER_API_KEY=your_key
   ```

3. Install & run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

## 🤖 Commands
| Command | Action |
|---------|--------|
| `/start` | Setup your stream |
| `/subject` | Choose focus subject |
| Ask anything | Get formatted answer |

## 🌍 Deployment
**Heroku** (Free Tier):
```bash
heroku create
git push heroku main
```

**PM2** (For VPS):
```bash
pm2 start app.py --name edumaster
```

## 📦 Requirements
```txt
python-telegram-bot==20.3
openai>=1.0
python-dotenv>=1.0
```

## 📌 Notes
• Uses DeepSeek model via OpenRouter  
• Optimized for Ethiopian curriculum  
• Supports group chats (mention bot + question)

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)
```