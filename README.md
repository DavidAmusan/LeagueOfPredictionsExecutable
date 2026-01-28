# League of Legends Champion Select Predictor

An intelligent real-time champion select monitor that uses machine learning to predict match outcomes and recommend optimal champion picks based on team compositions.

## Features

🤖 **AI-Powered Predictions**
- Predicts match outcome probability when both teams are complete
- Win percentage predictions for each team
- Trained on thousands of real match data

💡 **Smart Champion Recommendations**
- Recommends top 5 champions to maximize your team's win chance
- Updates automatically when enemy completes their picks
- Shows enemy's best picks to help you anticipate their choices

🎮 **Real-Time Monitoring**
- Automatically detects when you enter champion select
- Live updates as picks happen
- Clean, emoji-enhanced display
- Highlights which position is YOU

⚡ **State-Smart Updates**
- Only generates new predictions when champion selections change
- No spam - efficient and focused
- Shows match prediction in the final phase

## Requirements

### System Requirements
- macOS (League of Legends for Mac)
- Python 3.8+
- League of Legends client installed and running

### Python Dependencies
```bash
pip3 install tensorflow scikit-learn pandas numpy requests --break-system-packages
```

### Required Files
Place these files in the same directory as the script:

1. **games.csv** - Historical match data for training the AI model
   - Should contain columns for champion IDs, stats, and winners
   - Minimum ~10,000 games recommended for good predictions

2. **champion_info.json** - Champion ID to name mappings
   - Already provided in the project
   - Maps champion IDs to readable names

## Installation

1. **Install Python dependencies:**
```bash
pip3 install tensorflow scikit-learn pandas numpy requests --break-system-packages
```

2. **Place required files in the same directory:**
```
lol_ai_predictor.py
games.csv
champion_info.json
```

3. **Run the script:**
```bash
python3 lol_ai_predictor.py
```

## How It Works

### 1. Initial Startup
- Loads champion data (names and IDs)
- Trains AI model on historical game data (takes 2-5 minutes)
- Waits for League client to be detected

### 2. Champion Select Detection
- Automatically detects when you enter champion select
- Begins monitoring picks in real-time

### 3. During Pick Phase
- **Enemy picks complete, your team incomplete:**
  - Shows TOP 5 recommended champions for your team
  - Each recommendation includes win probability boost
  - Updates only when picks change (no spam)

- **Your team picks complete, enemy incomplete:**
  - Shows top 3 likely picks enemy will make
  - Helps you anticipate their strategy

### 4. Final Phase (Both Teams Complete)
- Shows final match outcome prediction
- Win percentages for both teams
- Advantage assessment (strong/even/disadvantage)

### 5. Post-Select
- Returns to waiting mode
- Ready for next champion select

## Understanding the Predictions

### Win Probability
- **60%+ for your team:** Strong advantage - draft favors you
- **40-60%:** Even matchup - skill will decide
- **<40% for your team:** Enemy has draft advantage - play carefully

### Champion Recommendations
- Listed from best to worst pick
- Each shows estimated win probability if you pick that champion
- Considers:
  - Team synergy
  - Enemy team composition
  - Historical performance data
  - Role balance

### Prediction Confidence
The AI is trained on real match data and considers:
- Champion winrates
- Team composition synergies
- Enemy matchups
- Historical patterns from thousands of games

**Note:** Predictions are probabilistic - player skill always matters most!

## Display Guide

### Symbols Used
- 🔝 Top lane
- 🌲 Jungle
- ⭐ Mid lane
- 🎯 Bot/ADC
- 💚 Support
- ✅ Locked in
- 👀 Hovering
- ⏳ Waiting
- ← YOU Your position

### Sections
1. **Timer** - Current phase and time remaining
2. **Your Team** - Your team's picks and roles
3. **Enemy Team** - Enemy picks and roles
4. **Bans** - Banned champions for both teams
5. **AI Predictions** - Smart recommendations and win predictions

## Example Output

```
================================================================================
                    🤖 LoL CHAMPION SELECT AI PREDICTOR
================================================================================
Updated: 14:32:15                         Press Ctrl+C to stop

📍 Phase: BAN_PICK
⏱️  Time Remaining: 28s

────────────────────────────────────────────────────────────────────────────────
🟦 YOUR TEAM
────────────────────────────────────────────────────────────────────────────────
🔝 [TOP     ] Garen                      ✅ LOCKED
🌲 [JUNGLE  ] Lee Sin                    ✅ LOCKED ← YOU
⭐ [MIDDLE  ] Ahri                       ✅ LOCKED
🎯 [BOTTOM  ] Jinx                       ✅ LOCKED
💚 [UTILITY ] Not selected               ⏳ Waiting

────────────────────────────────────────────────────────────────────────────────
🟥 ENEMY TEAM
────────────────────────────────────────────────────────────────────────────────
🔝 [TOP     ] Darius                     ✅ LOCKED
🌲 [JUNGLE  ] Master Yi                  ✅ LOCKED
⭐ [MIDDLE  ] Zed                        ✅ LOCKED
🎯 [BOTTOM  ] Caitlyn                    ✅ LOCKED
💚 [UTILITY ] Thresh                     ✅ LOCKED

────────────────────────────────────────────────────────────────────────────────
🚫 BANS
────────────────────────────────────────────────────────────────────────────────
Your Team:  Yasuo, Teemo, Blitzcrank
Enemy Team: Yuumi, Pyke, Morgana

================================================================================
🤖 AI PREDICTIONS
================================================================================

💡 CHAMPION RECOMMENDATIONS FOR YOUR TEAM:

   1. Leona                → 58.3% win chance
   2. Thresh                → 57.1% win chance
   3. Nautilus              → 56.8% win chance
   4. Braum                 → 56.2% win chance
   5. Alistar               → 55.7% win chance
================================================================================
```

## Troubleshooting

**"❌ Error: games.csv not found!"**
- Make sure games.csv is in the same directory as the script
- Download the training data and place it correctly

**"⚠️ Warning: Could not load champion info"**
- Check that champion_info.json is present
- The script will still work but show champion IDs instead of names

**"League of Legends client is not running"**
- Start the League client first
- The script will automatically detect it

**"Model training is slow"**
- First run trains the AI model (2-5 minutes)
- This is normal - subsequent runs are instant
- Consider using a smaller dataset for faster training during testing

**Predictions seem inaccurate**
- Ensure games.csv has sufficient data (10,000+ games)
- Check that the data is from recent patches
- Remember: predictions are probabilities, not guarantees!

## Technical Details

### AI Model Architecture
- Multi-task neural network
- Input: 10 champion IDs (5 per team)
- Outputs: Game stats predictions + winner probability
- Trained using TensorFlow/Keras

### Training Data
- Uses historical ranked game data
- Features: Champion compositions, objectives, winner
- Preprocessed to remove summoner spells and bans
- Standardized and normalized for optimal learning

### Prediction Method
For recommendations:
1. Generates all possible team compositions with available champions
2. Runs each through the trained model
3. Ranks by win probability
4. Returns top N recommendations

For win prediction:
1. Takes complete 5v5 composition
2. Predicts game statistics
3. Predicts winner based on stats + composition
4. Returns probability for each team

## Privacy & Safety

- **No data is sent anywhere** - all processing is local
- **No account access** - only reads publicly available match state
- **Read-only** - cannot make picks or change settings
- **Safe to use** - uses official League Client API

## Tips for Best Results

1. **Trust the AI for draft** - recommendations are data-driven
2. **Consider your skill** - AI shows optimal picks, but play what you're comfortable with
3. **Don't tilt on predictions** - a 40% chance means 4/10 games you still win!
4. **Use for learning** - see why certain picks are recommended
5. **Combine with game knowledge** - AI + your experience = best results

## Limitations

- Predictions based on historical data only
- Cannot account for:
  - Individual player skill
  - Current meta shifts
  - Patch changes not in training data
  - Team coordination
  - Player psychology
- Requires both teams to have 5 picks for full prediction
- Training data quality affects prediction quality

## Credits

- Built using League of Legends Client API (LCU)
- Machine learning powered by TensorFlow
- Champion data from Riot Games
- Historical match data from community datasets

## License

For educational and personal use. League of Legends and all related properties are owned by Riot Games.

---

**Good luck, and may the AI be with you!** 🎮✨
