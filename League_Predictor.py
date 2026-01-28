#!/usr/bin/env python3
"""
League of Legends Champion Select AI Predictor
Post-champion-select analysis with PMVP, objective predictions, and winner.

Requirements:
- games.csv (training data)
- champion_info.json (champion ID/name mappings)
- League of Legends client running

Usage: python3 League_Predictor_Updated.py
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
# MODEL TRAINING
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
    """
    from tensorflow import keras
    
    # Input layer
    input_layer = keras.layers.Input(shape=(input_dim,), name='champion_input')

    # Layer 1
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

    # Stats prediction branch
    stats_branch = keras.layers.Dense(
        stats_branch_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(shared)
    if use_batch_norm:
        stats_branch = keras.layers.BatchNormalization()(stats_branch)
    stats_branch = keras.layers.Dropout(dropout_rate * 0.5)(stats_branch)
    
    stats_output = keras.layers.Dense(
        output_stats_dim,
        activation='sigmoid',
        name='stats_output'
    )(stats_branch)

    # Winner prediction branch - concatenates shared + stats
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

    # Compile
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
    
    # Build and train model
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
        mode='min',
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

def predict_game_outcome(t1_champs, t2_champs, model, scaler, stat_features):
    """
    Predict winner and game statistics.
    Returns full prediction including stats for objectives.
    """
    t1_clean = [c for c in t1_champs if c and c != 0]
    t2_clean = [c for c in t2_champs if c and c != 0]
    
    if len(t1_clean) != 5 or len(t2_clean) != 5:
        return None
    
    try:
        game_input = np.array([t1_clean + t2_clean], dtype=np.float32)
        stats_pred, winner_pred = model.predict(game_input, verbose=0)
        
        # Unscale stats
        stats_unscaled = scaler.inverse_transform(stats_pred)[0]
        
        # Winner probabilities
        winner_prob = winner_pred[0][0]
        team1_prob = (1 - winner_prob) * 100
        team2_prob = winner_prob * 100
        
        # Map stats to dict
        stats_dict = {stat_features[i]: float(stats_unscaled[i]) for i in range(len(stat_features))}
        
        return {
            'team1_probability': team1_prob,
            'team2_probability': team2_prob,
            'predicted_winner': 1 if team1_prob > team2_prob else 2,
            'stats': stats_dict
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


def interpret_stat_prediction(value):
    """
    Interpret stat predictions into discrete outcomes.
    < 0.5: Contested
    0.5 - 1.0: Team gets 1
    1.0 - 1.5: Contested
    > 1.5: Team gets 2
    """
    if value < 0.5:
        return "Contested"
    elif value <= 1.0:
        return "1"
    elif value <= 1.5:
        return "Contested"
    else:
        return "2"


def calculate_pmvp(my_team, their_team, model, all_champions, id_to_name):
    """
    Calculate PMVP (Predicted Most Valuable Pick) for each team.
    Tests removing each champion and sees win rate impact.
    """
    def test_without_champion(team, opponent, exclude_idx):
        """Test win rate with one champion removed"""
        test_team = [c for i, c in enumerate(team) if i != exclude_idx]
        
        # Test with average champions
        available = [c for c in all_champions if c not in test_team and c not in opponent]
        if not available:
            return None
        
        results = []
        for replacement in available[:20]:  # Test top 20 available
            test_full = test_team[:exclude_idx] + [replacement] + test_team[exclude_idx:]
            game_input = np.array([test_full + opponent], dtype=np.float32)
            _, winner_pred = model.predict(game_input, verbose=0)
            team1_prob = (1 - winner_pred[0][0]) * 100
            results.append(team1_prob)
        
        return np.mean(results) if results else None
    
    # Calculate baseline
    game_input = np.array([my_team + their_team], dtype=np.float32)
    _, winner_pred = model.predict(game_input, verbose=0)
    baseline_my_team = (1 - winner_pred[0][0]) * 100
    baseline_their_team = 100 - baseline_my_team
    
    # Test each champion on my team
    my_impacts = []
    for i in range(5):
        without_prob = test_without_champion(my_team, their_team, i)
        if without_prob is not None:
            impact = baseline_my_team - without_prob  # Positive = champion increases win rate
            my_impacts.append({
                'champion_id': my_team[i],
                'champion_name': get_champion_name(my_team[i], id_to_name),
                'impact': impact
            })
    
    # Test each champion on their team
    their_impacts = []
    for i in range(5):
        # For enemy team, we need to flip perspective
        without_prob = test_without_champion(their_team, my_team, i)
        if without_prob is not None:
            # without_prob is their team's win rate without this champion
            impact = baseline_their_team - without_prob
            their_impacts.append({
                'champion_id': their_team[i],
                'champion_name': get_champion_name(their_team[i], id_to_name),
                'impact': impact
            })
    
    # Find MVPs
    my_mvp = max(my_impacts, key=lambda x: x['impact']) if my_impacts else None
    their_mvp = max(their_impacts, key=lambda x: x['impact']) if their_impacts else None
    
    return my_mvp, their_mvp


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def clear_screen():
    """Clear terminal screen"""
    os.system('clear')


class PredictionDisplay:
    """Manages persistent prediction display"""
    def __init__(self):
        self.predictions = None
        self.my_team = []
        self.their_team = []
        self.my_team_names = []
        self.their_team_names = []
    
    def set_predictions(self, predictions, my_team, their_team, my_team_names, their_team_names):
        """Store predictions to display persistently"""
        self.predictions = predictions
        self.my_team = my_team
        self.their_team = their_team
        self.my_team_names = my_team_names
        self.their_team_names = their_team_names
    
    def clear(self):
        """Clear predictions for new session"""
        self.predictions = None
        self.my_team = []
        self.their_team = []
        self.my_team_names = []
        self.their_team_names = []
    
    def display(self, phase="", time_left=0):
        """Display predictions"""
        clear_screen()
        print("=" * 80)
        print(" " * 20 + "🤖 LoL CHAMPION SELECT AI PREDICTOR")
        print("=" * 80)
        print(f"Updated: {time.strftime('%H:%M:%S')}" + " " * 25 + "Press Ctrl+C to stop")
        print()
        if phase:
            print(f"📍 Phase: {phase}")
            print(f"⏱️  Time Remaining: {int(time_left)}s")
        print()
        
        if not self.predictions:
            print("⏳ Waiting for champion select to complete...\n")
            return
        
        pred = self.predictions
        
        # Display teams
        print("─" * 80)
        print("🟦 YOUR TEAM")
        print("─" * 80)
        for name in self.my_team_names:
            print(f"  • {name}")
        
        print()
        print("─" * 80)
        print("🟥 ENEMY TEAM")
        print("─" * 80)
        for name in self.their_team_names:
            print(f"  • {name}")
        
        # Predictions
        print()
        print("=" * 80)
        print("📊 MATCH PREDICTIONS")
        print("=" * 80)
        
        # Winner
        winner_team = "Your Team" if pred['predicted_winner'] == 1 else "Enemy Team"
        team1_prob = pred['team1_probability']
        team2_prob = pred['team2_probability']
        
        print(f"\n🏆 PREDICTED WINNER: {winner_team}")
        print(f"   Your Team:  {team1_prob:.1f}%")
        print(f"   Enemy Team: {team2_prob:.1f}%")
        
        diff = abs(team1_prob - team2_prob)
        if diff >= 15:
            if team1_prob > team2_prob:
                print(f"   ✨ Strong advantage for your team!")
            else:
                print(f"   ⚠️  Enemy team has strong advantage")
        elif diff >= 8:
            if team1_prob > team2_prob:
                print(f"   💪 Your team is favored")
            else:
                print(f"   😬 Enemy team is favored")
        else:
            print(f"   ⚖️  Even matchup!")
        
        # PMVP
        if 'my_mvp' in pred and pred['my_mvp']:
            print(f"\n⭐ YOUR TEAM PMVP: {pred['my_mvp']['champion_name']}")
            print(f"   Impact: +{pred['my_mvp']['impact']:.1f}% win rate")
        
        if 'their_mvp' in pred and pred['their_mvp']:
            print(f"\n⭐ ENEMY TEAM PMVP: {pred['their_mvp']['champion_name']}")
            print(f"   Impact: +{pred['their_mvp']['impact']:.1f}% win rate")
        
        # Objectives
        stats = pred['stats']
        
        print(f"\n🎯 OBJECTIVE PREDICTIONS")
        print(f"   (Based on predicted game statistics)")
        
        # First Tower
        t1_towers = stats.get('t1_towerKills', 0)
        t2_towers = stats.get('t2_towerKills', 0)
        first_tower_t1 = interpret_stat_prediction(t1_towers)
        first_tower_t2 = interpret_stat_prediction(t2_towers)
        
        if first_tower_t1 == "Contested" or first_tower_t2 == "Contested":
            print(f"\n🏰 First Turret: Contested")
        elif first_tower_t1 != "Contested" and float(first_tower_t1) > 0:
            print(f"\n🏰 First Turret: Your Team (predicted {first_tower_t1} towers)")
        elif first_tower_t2 != "Contested" and float(first_tower_t2) > 0:
            print(f"\n🏰 First Turret: Enemy Team (predicted {first_tower_t2} towers)")
        else:
            print(f"\n🏰 First Turret: Contested")
        
        # First Dragon
        t1_dragons = stats.get('t1_dragonKills', 0)
        t2_dragons = stats.get('t2_dragonKills', 0)
        first_dragon_t1 = interpret_stat_prediction(t1_dragons)
        first_dragon_t2 = interpret_stat_prediction(t2_dragons)
        
        if first_dragon_t1 == "Contested" or first_dragon_t2 == "Contested":
            print(f"🐉 First Dragon: Contested")
        elif first_dragon_t1 != "Contested" and float(first_dragon_t1) > 0:
            print(f"🐉 First Dragon: Your Team (predicted {first_dragon_t1} dragons)")
        elif first_dragon_t2 != "Contested" and float(first_dragon_t2) > 0:
            print(f"🐉 First Dragon: Enemy Team (predicted {first_dragon_t2} dragons)")
        else:
            print(f"🐉 First Dragon: Contested")
        
        # First Blood (approximation based on general aggression)
        # We'll use a simple heuristic: team with higher predicted stats is more aggressive
        t1_aggression = t1_towers + t1_dragons + stats.get('t1_riftHeraldKills', 0)
        t2_aggression = t2_towers + t2_dragons + stats.get('t2_riftHeraldKills', 0)
        
        if abs(t1_aggression - t2_aggression) < 0.5:
            print(f"💀 First Blood: Contested")
        elif t1_aggression > t2_aggression:
            print(f"💀 First Blood: Your Team (higher predicted aggression)")
        else:
            print(f"💀 First Blood: Enemy Team (higher predicted aggression)")
        
        print()
        print("=" * 80)


def display_session(session, id_to_name, model, scaler, stat_features, all_champions, 
                    prediction_display, last_state):
    """Monitor champion select and make predictions when complete"""
    
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
    their_team_locked = []
    
    for player in my_team:
        champion_id = player.get('championId', 0)
        if champion_id > 0:
            my_team_locked.append(champion_id)
    
    for player in their_team:
        champion_id = player.get('championId', 0)
        if champion_id > 0:
            their_team_locked.append(champion_id)
    
    # Create state for change detection
    current_state = {
        'my_locked': tuple(sorted(my_team_locked)),
        'their_locked': tuple(sorted(their_team_locked)),
        'phase': phase
    }
    
    # Check if both teams are complete (10 champions total)
    if len(my_team_locked) == 5 and len(their_team_locked) == 5:
        # Make predictions if not already done for this composition
        if current_state != last_state:
            # Get predictions
            prediction = predict_game_outcome(my_team_locked, their_team_locked, model, scaler, stat_features)
            
            if prediction:
                # Calculate PMVP
                my_mvp, their_mvp = calculate_pmvp(my_team_locked, their_team_locked, model, all_champions, id_to_name)
                
                prediction['my_mvp'] = my_mvp
                prediction['their_mvp'] = their_mvp
                
                # Get champion names
                my_team_names = [get_champion_name(c, id_to_name) for c in my_team_locked]
                their_team_names = [get_champion_name(c, id_to_name) for c in their_team_locked]
                
                # Store predictions
                prediction_display.set_predictions(
                    prediction, 
                    my_team_locked, 
                    their_team_locked,
                    my_team_names,
                    their_team_names
                )
    
    # Display (will show predictions if available)
    prediction_display.display(phase, time_left)
    
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
    
    # Train model in a separate thread
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
    prediction_display = PredictionDisplay()
    
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
                    # NEW CHAMPION SELECT - Clear predictions
                    prediction_display.clear()
                    if model_data['ready']:
                        print("\n🎮 Entered Champion Select - Waiting for picks...\n")
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
                        prediction_display, last_state
                    )
            else:
                if in_champ_select:
                    # Left champ select - keep showing predictions
                    in_champ_select = False
                
                # Keep displaying predictions even outside champ select
                if prediction_display.predictions:
                    prediction_display.display()
                
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
