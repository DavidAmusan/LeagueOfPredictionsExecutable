#!/usr/bin/env python3
"""
League of Legends Champion Select AI Predictor
Combines real-time champion select monitoring with ML-based win prediction
and champion recommendations.

Requirements:
- games.csv (training data)
- champion_info.json (champion ID/name mappings)
- League of Legends client running

Usage: python3 lol_ai_predictor.py
"""

import requests
import base64
import time
import os
import sys
import json
import numpy as np
import pandas as pd
import threading
from urllib3.exceptions import InsecureRequestWarning
from pathlib import Path

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Configuration
LOCKFILE_PATH = "/Applications/League of Legends.app/Contents/LoL/lockfile"
UPDATE_INTERVAL = 1.5
GAMES_CSV = "games.csv"
CHAMPION_JSON = "champion_info.json"

# ============================================================
# LEAGUE CLIENT CONNECTION
# ============================================================

class LCUClient:
    """League Client API connection handler"""
    
    def __init__(self):
        self.base_url = None
        self.headers = None
        self.connected = False
    
    def connect(self):
        """Read lockfile and establish connection credentials"""
        try:
            with open(LOCKFILE_PATH, 'r') as f:
                content = f.read()
            
            parts = content.split(':')
            if len(parts) < 5:
                return False
            
            port = parts[2]
            password = parts[3]
            protocol = parts[4]
            
            self.base_url = f"{protocol}://127.0.0.1:{port}"
            auth_string = f"riot:{password}"
            auth_bytes = base64.b64encode(auth_string.encode('ascii'))
            
            self.headers = {
                'Authorization': f'Basic {auth_bytes.decode("ascii")}',
                'Content-Type': 'application/json'
            }
            
            self.connected = True
            return True
            
        except FileNotFoundError:
            self.connected = False
            return False
        except Exception:
            self.connected = False
            return False
    
    def get(self, endpoint):
        """Make GET request to LCU API"""
        if not self.connected:
            return None
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                verify=False,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None


# ============================================================
# CHAMPION MAPPING
# ============================================================

def load_champion_mapping(champion_info_file=CHAMPION_JSON):
    """Load champion ID to name mappings"""
    try:
        with open(champion_info_file, 'r') as f:
            champion_data = json.load(f)
        
        id_to_name = {}
        name_to_id = {}
        
        if 'data' in champion_data:
            champ_dict = champion_data['data']
        else:
            champ_dict = champion_data
        
        for key, champ_info in champ_dict.items():
            if isinstance(champ_info, dict):
                champ_id = champ_info.get('id') or int(key)
                champ_name = champ_info.get('name')
                
                if champ_id and champ_name:
                    id_to_name[int(champ_id)] = champ_name
                    name_to_id[champ_name.lower()] = int(champ_id)
        
        return id_to_name, name_to_id
    except Exception as e:
        print(f"⚠️  Warning: Could not load champion info: {e}")
        return {}, {}


def get_champion_name(champion_id, id_to_name):
    """Convert champion ID to name"""
    if champion_id == 0 or champion_id == -1:
        return None
    return id_to_name.get(champion_id, f"Champion {champion_id}")


# ============================================================
# MODEL TRAINING (EXACT ORIGINAL ARCHITECTURE)
# ============================================================

def build_optimized_model(
    input_dim,
    output_stats_dim,
    layer1_units=320,
    layer2_units=160,
    layer3_units=80,
    layer4_units=40,
    stats_branch_units=80,
    winner_branch1_units=120,
    winner_branch2_units=60,
    dropout_rate=0.28,
    l2_reg=0.0012,
    use_batch_norm=True,
    stats_weight=0.12,
    winner_weight=0.88,
    learning_rate=0.0012
):
    """
    EXACT model from LegendModels.ipynb - WinnerPrimary configuration
    Architecture with 5 layers (4 shared + stats/winner branches)
    Uses stats concatenation in winner branch for better predictions.
    
    Key differences from generic model:
    - Stats output uses SIGMOID activation (for binary/normalized stats)
    - Winner branch CONCATENATES shared layer + predicted stats
    - Binary crossentropy for stats loss (not MAE)
    - Winner weight = 0.88 (heavily favors winner prediction)
    """
    from tensorflow import keras
    
    # Input layer
    input_layer = keras.layers.Input(shape=(input_dim,), name='champion_input')

    # Layer 1: Deep shared trunk
    x = keras.layers.Dense(
        layer1_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(input_layer)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Layer 2
    x = keras.layers.Dense(
        layer2_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Layer 3
    x = keras.layers.Dense(
        layer3_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate * 0.75)(x)

    # Layer 4: Final shared layer
    shared = keras.layers.Dense(
        layer4_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    if use_batch_norm:
        shared = keras.layers.BatchNormalization()(shared)
    shared = keras.layers.Dropout(dropout_rate * 0.5)(shared)

    # Stats prediction branch (Layer 5a)
    stats_branch = keras.layers.Dense(
        stats_branch_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(shared)
    if use_batch_norm:
        stats_branch = keras.layers.BatchNormalization()(stats_branch)
    stats_branch = keras.layers.Dropout(dropout_rate * 0.5)(stats_branch)
    
    # CRITICAL: Sigmoid activation for binary/normalized stats
    stats_output = keras.layers.Dense(
        output_stats_dim,
        activation='sigmoid',
        name='stats_output'
    )(stats_branch)

    # Winner prediction branch (Layer 5b)
    # CRITICAL: Concatenate shared features WITH predicted stats
    winner_input = keras.layers.Concatenate()([shared, stats_output])
    
    winner_branch = keras.layers.Dense(
        winner_branch1_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(winner_input)
    if use_batch_norm:
        winner_branch = keras.layers.BatchNormalization()(winner_branch)
    winner_branch = keras.layers.Dropout(dropout_rate * 0.5)(winner_branch)
    
    winner_branch = keras.layers.Dense(
        winner_branch2_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(winner_branch)
    winner_branch = keras.layers.Dropout(dropout_rate * 0.3)(winner_branch)
    
    winner_output = keras.layers.Dense(
        1,
        activation='sigmoid',
        name='winner_output'
    )(winner_branch)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=[stats_output, winner_output])

    # Compile with multi-task loss
    # CRITICAL: Binary crossentropy for both outputs (stats are 0/1)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'stats_output': 'binary_crossentropy',
            'winner_output': 'binary_crossentropy'
        },
        loss_weights={
            'stats_output': stats_weight,
            'winner_output': winner_weight
        },
        metrics={
            'stats_output': ['accuracy', 'mae'],
            'winner_output': ['accuracy', keras.metrics.AUC(name='auc'),
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall')]
        }
    )

    return model


def load_and_train_model():
    """Train the model from CSV data"""
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print(f"📊 Loading training data from {GAMES_CSV}...")
    games_df = pd.read_csv(GAMES_CSV)
    
    # Define features
    champion_features = [f't{t}_champ{i}id' for t in [1, 2] for i in range(1, 6)]
    stat_features = [
        't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills',
        't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills'
    ]
    winner_feature = 'winner'
    
    # Clean data
    games_df_clean = games_df.dropna(subset=champion_features + stat_features + [winner_feature])
    games_df_clean = games_df_clean[
        (games_df_clean[champion_features] > 0).all(axis=1)
    ]
    
    print(f"✓ Loaded {len(games_df_clean)} games")
    
    # Prepare data
    X = games_df_clean[champion_features].values.astype(np.float32)
    y_stats = games_df_clean[stat_features].values.astype(np.float32)
    y_winner = (games_df_clean[winner_feature].values == 2).astype(np.float32)
    
    # Normalize stats
    from sklearn.preprocessing import StandardScaler
    scaler_stats = StandardScaler()
    y_stats_scaled = scaler_stats.fit_transform(y_stats)
    
    # Build and train model with WinnerPrimary configuration
    print("🧠 Training AI model (WinnerPrimary config)...")
    model = build_optimized_model(
        input_dim=len(champion_features),
        output_stats_dim=len(stat_features),
        layer1_units=320,
        layer2_units=160,
        layer3_units=80,
        layer4_units=40,
        stats_branch_units=80,
        winner_branch1_units=120,
        winner_branch2_units=60,
        dropout_rate=0.28,
        l2_reg=0.0012,
        use_batch_norm=True,
        stats_weight=0.12,
        winner_weight=0.88,
        learning_rate=0.0012
    )
    
    # Train with early stopping
    from tensorflow import keras
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_winner_output_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_winner_output_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=0
    )
    
    model.fit(
        X,
        {'stats_output': y_stats_scaled, 'winner_output': y_winner},
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Get all unique champions
    all_champions = set()
    for feature in champion_features:
        all_champions.update(games_df_clean[feature].unique())
    all_champions = sorted([int(c) for c in all_champions if c > 0 and not pd.isna(c)])
    
    return model, scaler_stats, stat_features, all_champions


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_winner(t1_champs, t2_champs, model, scaler, stat_features):
    """Predict winner given full team compositions"""
    # Clean the input lists - remove any None or 0 values
    t1_clean = [c for c in t1_champs if c and c != 0]
    t2_clean = [c for c in t2_champs if c and c != 0]
    
    # Check if we have valid 5v5
    if len(t1_clean) != 5 or len(t2_clean) != 5:
        return None
    
    try:
        game_input = np.array([t1_clean + t2_clean], dtype=np.float32)
        stats_pred, winner_pred = model.predict(game_input, verbose=0)
        
        winner_prob = winner_pred[0][0]
        team1_prob = (1 - winner_prob) * 100
        team2_prob = winner_prob * 100
        
        return {
            'team1_probability': team1_prob,
            'team2_probability': team2_prob,
            'predicted_winner': 1 if team1_prob > team2_prob else 2
        }
    except Exception as e:
        print(f"Error in predict_winner: {e}")
        return None



def recommend_champion_for_team(team1_partial, team2_partial, model, all_champions, top_n=5):
    """
    Recommend champions for incomplete team.
    FIXED: Now handles cases where BOTH teams are incomplete.
    """
    # Clean the inputs
    team1_current = [c for c in team1_partial if c and c != 0]
    team2_current = [c for c in team2_partial if c and c != 0]
    
    # Can't recommend if team1 is already full
    if len(team1_current) >= 5:
        return []
    
    # Calculate missing slots for both teams
    team1_missing = 5 - len(team1_current)
    team2_missing = 5 - len(team2_current)
    
    # Get already picked champions
    already_picked = set(team1_current + team2_current)
    available_champions = [c for c in all_champions if c not in already_picked]
    
    # Pad both teams to 5 champions
    team1_base = team1_current + [0] * team1_missing
    team2_base = team2_current + [0] * team2_missing
    
    results = []
    
    try:
        for champion_id in available_champions:
            # Test adding this champion to team1
            team1_test = team1_base.copy()
            first_empty_idx = team1_test.index(0)
            team1_test[first_empty_idx] = champion_id
            
            # Create full game input
            game_input = np.array([team1_test + team2_base], dtype=np.float32)
            _, winner_pred = model.predict(game_input, verbose=0)
            
            # Team1 win probability
            team1_win_prob = (1 - winner_pred[0][0]) * 100
            
            results.append({
                'champion_id': champion_id,
                'win_probability': team1_win_prob
            })
    except Exception as e:
        print(f"Error in recommend_champion_for_team: {e}")
        return []
    
    results.sort(key=lambda x: x['win_probability'], reverse=True)
    return results[:top_n]


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def clear_screen():
    """Clear terminal screen"""
    os.system('clear')


class PredictionLog:
    """Manages persistent prediction log"""
    def __init__(self):
        self.logs = []
        self.current_session_id = None
        self.predictions_made = set()  # Track which predictions we've made
    
    def new_session(self):
        """Start a new champion select session"""
        self.logs = []
        self.predictions_made = set()
        self.current_session_id = time.time()
    
    def add_prediction(self, message, prediction_key=None):
        """Add a prediction to the log"""
        # Prevent duplicate predictions for the same stage
        if prediction_key and prediction_key in self.predictions_made:
            return
        
        timestamp = time.strftime('%H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        
        if prediction_key:
            self.predictions_made.add(prediction_key)
    
    def display(self):
        """Display all logged predictions"""
        if self.logs:
            print("\n" + "=" * 80)
            print("📜 PREDICTION HISTORY")
            print("=" * 80)
            for log in self.logs:
                print(log)
            print("=" * 80)


def display_session(session, id_to_name, model, scaler, stat_features, all_champions, 
                    prediction_log, last_state):
    """Display formatted champion select session with predictions"""
    
    # Extract team data
    my_team = session.get('myTeam', [])
    their_team = session.get('theirTeam', [])
    local_player_cell_id = session.get('localPlayerCellId', -1)
    
    # Timer
    timer = session.get('timer', {})
    phase = timer.get('phase', 'Unknown')
    time_left = timer.get('adjustedTimeLeftInPhase', 0) / 1000
    
    # Build team lists
    my_team_locked = []
    my_team_hovering = []
    their_team_locked = []
    their_team_hovering = []
    
    for player in my_team:
        champion_id = player.get('championId', 0)
        intent_id = player.get('championPickIntent', 0)
        
        if champion_id > 0:
            my_team_locked.append(champion_id)
        elif intent_id > 0:
            my_team_hovering.append(intent_id)
    
    for player in their_team:
        champion_id = player.get('championId', 0)
        intent_id = player.get('championPickIntent', 0)
        
        if champion_id > 0:
            their_team_locked.append(champion_id)
        elif intent_id > 0:
            their_team_hovering.append(intent_id)
    
    # Create state for change detection
    current_state = {
        'my_locked': tuple(sorted(my_team_locked)),
        'my_hovering': tuple(sorted(my_team_hovering)),
        'their_locked': tuple(sorted(their_team_locked)),
        'their_hovering': tuple(sorted(their_team_hovering)),
        'phase': phase
    }
    
    # Display header (always)
    clear_screen()
    print("=" * 80)
    print(" " * 20 + "🤖 LoL CHAMPION SELECT AI PREDICTOR")
    print("=" * 80)
    print(f"Updated: {time.strftime('%H:%M:%S')}" + " " * 25 + "Press Ctrl+C to stop")
    print()
    print(f"📍 Phase: {phase}")
    print(f"⏱️  Time Remaining: {int(time_left)}s")
    print()
    
    # Display teams
    print("─" * 80)
    print("🟦 YOUR TEAM")
    print("─" * 80)
    
    for player in sorted(my_team, key=lambda x: x.get('cellId', 0)):
        cell_id = player.get('cellId', -1)
        champion_id = player.get('championId', 0)
        intent_id = player.get('championPickIntent', 0)
        position = player.get('assignedPosition', 'FILL')
        
        position_emoji = {
            'top': '🔝', 'jungle': '🌲', 'middle': '⭐',
            'bottom': '🎯', 'utility': '💚'
        }.get(position.lower(), '❓')
        
        is_you = (cell_id == local_player_cell_id)
        you_marker = " ← YOU" if is_you else ""
        
        if champion_id > 0:
            champ_name = get_champion_name(champion_id, id_to_name)
            status = "✅ LOCKED"
            display_name = champ_name
        elif intent_id > 0:
            champ_name = get_champion_name(intent_id, id_to_name)
            status = "👀 Hovering"
            display_name = champ_name
        else:
            display_name = "Not selected"
            status = "⏳ Waiting"
        
        print(f"{position_emoji} [{position.upper():8}] {display_name:25} {status}{you_marker}")
    
    if their_team:
        print()
        print("─" * 80)
        print("🟥 ENEMY TEAM")
        print("─" * 80)
        
        for player in sorted(their_team, key=lambda x: x.get('cellId', 0)):
            champion_id = player.get('championId', 0)
            intent_id = player.get('championPickIntent', 0)
            position = player.get('assignedPosition', 'UNKNOWN')
            
            position_emoji = {
                'top': '🔝', 'jungle': '🌲', 'middle': '⭐',
                'bottom': '🎯', 'utility': '💚'
            }.get(position.lower(), '❓')
            
            if champion_id > 0:
                champ_name = get_champion_name(champion_id, id_to_name)
                display_name = champ_name
                status = "✅ LOCKED"
            elif intent_id > 0:
                champ_name = get_champion_name(intent_id, id_to_name)
                display_name = champ_name
                status = "👀 Hovering"
            else:
                display_name = "Not selected"
                status = "⏳ Waiting"
            
            print(f"{position_emoji} [{position.upper():8}] {display_name:25} {status}")
    
    # Bans
    bans = session.get('bans', {})
    my_team_bans = bans.get('myTeamBans', [])
    their_team_bans = bans.get('theirTeamBans', [])
    
    if my_team_bans or their_team_bans:
        print()
        print("─" * 80)
        print("🚫 BANS")
        print("─" * 80)
        
        my_bans_str = [get_champion_name(b, id_to_name) for b in my_team_bans if b > 0]
        their_bans_str = [get_champion_name(b, id_to_name) for b in their_team_bans if b > 0]
        
        print(f"Your Team:  {', '.join(my_bans_str) if my_bans_str else 'None yet'}")
        print(f"Enemy Team: {', '.join(their_bans_str) if their_bans_str else 'None yet'}")
    
    # AI PREDICTIONS - Generate new predictions if state changed
    if current_state != last_state:
        # Track total picks to determine draft stage
        total_my_picks = len(my_team_locked)
        total_their_picks = len(their_team_locked)
        total_picks = total_my_picks + total_their_picks
        
        # Both teams have 5 locked - FINAL prediction (Prediction #6)
        if total_my_picks == 5 and total_their_picks == 5:
            prediction = predict_winner(my_team_locked, their_team_locked, model, scaler, stat_features)
            if prediction:
                team1_prob = prediction['team1_probability']
                team2_prob = prediction['team2_probability']
                
                msg = f"🎯 FINAL PREDICTION: Your Team {team1_prob:.1f}% | Enemy {team2_prob:.1f}%"
                prediction_log.add_prediction(msg, prediction_key="final")
                
                # Better thresholds for matchup evaluation
                diff = abs(team1_prob - team2_prob)
                if diff >= 15:  # 65-35 or more extreme
                    if team1_prob > team2_prob:
                        prediction_log.add_prediction("   ✨ Strong advantage for your team!")
                    else:
                        prediction_log.add_prediction("   ⚠️  Enemy team has the advantage")
                elif diff >= 8:  # 58-42 to 65-35
                    if team1_prob > team2_prob:
                        prediction_log.add_prediction("   💪 Your team is favored")
                    else:
                        prediction_log.add_prediction("   😬 Enemy team is favored")
                else:  # Less than 58-42
                    prediction_log.add_prediction("   ⚖️  Even matchup!")
        
        # During picking phase - predict based on League's 1-2-2-2-2-1 format
        # Blue side (your team): You(1) -> Enemy(2,3) -> You(4,5) -> Enemy(6,7) -> You(8,9) -> Enemy(10)
        # Red side (enemy team): Enemy(1) -> You(2,3) -> Enemy(4,5) -> You(6,7) -> Enemy(8,9) -> You(10)
        elif total_picks > 0 and total_picks < 10:
            should_predict = False
            predict_for_enemy = False
            prediction_stage = None
            
            # After 3 total picks (1-2-2 pattern complete) - Prediction #1
            if total_picks == 3:
                should_predict = True
                prediction_stage = "stage_3"
                # If your team has 1 pick, enemy picked 2-3, recommend for your team (picks 4-5)
                # If enemy has 1 pick, you picked 2-3, predict enemy (picks 4-5)
                if total_my_picks == 1:
                    predict_for_enemy = False
                elif total_their_picks == 1:
                    predict_for_enemy = True
            
            # After 5 total picks (1-2-2-2 pattern complete) - Prediction #2
            elif total_picks == 5:
                should_predict = True
                prediction_stage = "stage_5"
                # Whoever has 3 picks just finished their 4-5 picks
                if total_my_picks == 3:
                    predict_for_enemy = True  # Predict enemy (picks 6-7)
                elif total_their_picks == 3:
                    predict_for_enemy = False  # Recommend for your team (picks 6-7)
            
            # After 7 total picks (1-2-2-2-2 pattern complete) - Prediction #3
            elif total_picks == 7:
                should_predict = True
                prediction_stage = "stage_7"
                # Whoever has 4 picks just finished their 6-7 picks
                if total_my_picks == 4:
                    predict_for_enemy = True  # Predict enemy (picks 8-9)
                elif total_their_picks == 4:
                    predict_for_enemy = False  # Recommend for your team (picks 8-9)
            
            # After 9 total picks (1-2-2-2-2-1 pattern complete) - Prediction #4
            elif total_picks == 9:
                should_predict = True
                prediction_stage = "stage_9"
                # Whoever has 5 picks just finished their 8-9 picks
                if total_my_picks == 5:
                    predict_for_enemy = True  # Predict enemy (pick 10)
                elif total_their_picks == 5:
                    predict_for_enemy = False  # Recommend for your team (pick 10)
            
            # Make the prediction
            if should_predict and prediction_stage:
                if predict_for_enemy:
                    # Predict what enemy will pick
                    their_team_partial = their_team_locked + their_team_hovering
                    my_team_partial = my_team_locked
                    
                    recommendations = recommend_champion_for_team(
                        their_team_partial, my_team_partial, model, all_champions, top_n=3
                    )
                    
                    if recommendations:
                        prediction_log.add_prediction(
                            f"👁️  ENEMY LIKELY TO PICK:",
                            prediction_key=prediction_stage
                        )
                        for i, rec in enumerate(recommendations[:3], 1):
                            champ_name = get_champion_name(rec['champion_id'], id_to_name)
                            prediction_log.add_prediction(
                                f"   {i}. {champ_name:20} → {rec['win_probability']:.1f}% (for them)"
                            )
                
                else:
                    # Recommend for your team
                    my_team_partial = my_team_locked + my_team_hovering
                    their_team_partial = their_team_locked
                    
                    recommendations = recommend_champion_for_team(
                        my_team_partial, their_team_partial, model, all_champions, top_n=5
                    )
                    
                    if recommendations:
                        prediction_log.add_prediction(
                            f"💡 RECOMMENDATIONS FOR YOUR TEAM:",
                            prediction_key=prediction_stage
                        )
                        for i, rec in enumerate(recommendations[:5], 1):
                            champ_name = get_champion_name(rec['champion_id'], id_to_name)
                            prediction_log.add_prediction(
                                f"   {i}. {champ_name:20} → {rec['win_probability']:.1f}%"
                            )

    
    # Display persistent log
    prediction_log.display()
    
    return current_state


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("🤖 League of Legends Champion Select AI Predictor")
    print("=" * 80)
    print()
    
    # Load champion mappings
    print("📖 Loading champion data...")
    id_to_name, name_to_id = load_champion_mapping()
    print(f"✓ Loaded {len(id_to_name)} champions")
    
    # Train model in a separate thread for speed
    print("🚀 Starting model training in background thread...")
    model_data = {'model': None, 'scaler': None, 'stat_features': None, 'all_champions': None, 'ready': False}
    
    def train_in_thread():
        model, scaler, stat_features, all_champions = load_and_train_model()
        model_data['model'] = model
        model_data['scaler'] = scaler
        model_data['stat_features'] = stat_features
        model_data['all_champions'] = all_champions
        model_data['ready'] = True
        print("\n✨ Model training complete! AI predictions now active.\n")
    
    training_thread = threading.Thread(target=train_in_thread, daemon=True)
    training_thread.start()
    
    client = LCUClient()
    in_champ_select = False
    last_state = None
    prediction_log = PredictionLog()
    
    print()
    print("✓ Ready! Waiting for League of Legends client...")
    print("  (AI predictions will activate once model training completes)")
    print()
    
    try:
        while True:
            if not client.connected:
                if client.connect():
                    print("✓ Connected to League Client")
                else:
                    if in_champ_select:
                        print("\n⚠️  League client closed")
                        in_champ_select = False
                    time.sleep(2)
                    continue
            
            session = client.get('/lol-champ-select/v1/session')
            
            if session:
                if not in_champ_select:
                    # NEW CHAMPION SELECT - Clear everything
                    prediction_log.new_session()
                    if model_data['ready']:
                        print("\n🎮 Entered Champion Select - Starting AI predictions...\n")
                    else:
                        print("\n🎮 Entered Champion Select - Model still training...\n")
                    time.sleep(1)
                    in_champ_select = True
                    last_state = None
                
                # Only make predictions if model is ready
                if model_data['ready']:
                    last_state = display_session(
                        session, id_to_name, 
                        model_data['model'], model_data['scaler'], 
                        model_data['stat_features'], model_data['all_champions'], 
                        prediction_log, last_state
                    )
            else:
                if in_champ_select:
                    # Keep showing predictions after champ select ends
                    # Don't clear - let user see final predictions
                    in_champ_select = False
                
                summoner = client.get('/lol-summoner/v1/current-summoner')
                if not summoner:
                    client.connected = False
            
            time.sleep(UPDATE_INTERVAL)
    
    except KeyboardInterrupt:
        clear_screen()
        print("\n" + "=" * 80)
        print("🤖 AI Predictor stopped. Good luck in your games!")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
