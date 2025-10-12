# The Times of AI

AI-powered news aggregation platform that curates the latest AI and technology news using multi-agent intelligence.

## 🚀 Live Demo

**[https://sanjayhjp121.github.io/the-times-of-ai](https://sanjayhjp121.github.io/the-times-of-ai)**

## How It Works

1. **Collects** news from 30+ RSS feeds and APIs
2. **Processes** articles using AI agents (Groq models)
3. **Filters** content through consensus algorithms
4. **Generates** curated news feed and audio podcast
5. **Deploys** to GitHub Pages automatically

## Features

- 🤖 **AI Curation** - Multi-agent processing with fact-checking
- 📰 **Smart Classification** - Headlines, articles, and research papers
- 🎵 **Audio Podcast** - AI-generated news podcast
- 📱 **Responsive Design** - Works on all devices
- ⚡ **Serverless** - Runs on GitHub Actions + Pages (free)

## Local Development

```bash
# Clone and setup
git clone https://github.com/sanjayhjp121/the-times-of-ai.git
cd ai-newspaper

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r src/backend/requirements.txt
npm install

# Set up API keys
echo "GROQ_API_KEY=your_key_here" > .env.local
echo "GEMINI_API_KEY=your_key_here" >> .env.local

# Run the pipeline
./orchestrator.run

# Start frontend
npm run dev
```

## Configuration

- **News Sources**: `src/shared/config/sources/*.yaml`
- **AI Models**: `src/shared/config/swarm.yaml`
- **Pipeline Settings**: `src/shared/config/app.yaml`
