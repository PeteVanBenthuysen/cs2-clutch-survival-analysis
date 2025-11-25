"""
Audio tracking system for post-plant analysis.
Detects what sounds each player could hear and from whom.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.zone_classifier import MirageZoneClassifier


class AudioTracker:
    """Track audio cues that players can hear during post-plant."""
    
    def __init__(self):
        self.classifier = MirageZoneClassifier()
    
    def parse_audio_events(self, parser, plant_tick: int, retake_tick: int, team_filter: int = 2, 
                          start_tick: int = None, end_tick: int = None) -> Dict:
        """
        Parse all audio events and determine what each player could hear.
        
        Args:
            parser: DemoParser instance
            plant_tick: Tick when bomb was planted
            retake_tick: Tick when retake started
            team_filter: Team number to analyze (2=T, 3=CT). Analyze what this team heard.
            start_tick: Override start of window (default: plant_tick)
            end_tick: Override end of window (default: retake_tick)
        
        Returns dict with:
        - player_sounds_heard: Dict[player_name] -> List of sounds they heard
        - ct_sound_events: All CT sounds that Ts could hear
        - audio_timeline: Chronological list of all audible events
        """
        
        # Allow custom window for pre-plant analysis
        window_start = start_tick if start_tick is not None else plant_tick
        window_end = end_tick if end_tick is not None else retake_tick
        
        # Parse sound events
        player_sounds = parser.parse_event('player_sound')
        
        # Parse player positions
        ticks = parser.parse_ticks(['X', 'Y', 'Z', 'name', 'tick', 'team_num', 'steamid'])
        
        # Filter to analysis window
        window_sounds = player_sounds[
            (player_sounds['tick'] >= window_start) & 
            (player_sounds['tick'] <= window_end)
        ]
        
        print(f"\nProcessing {len(window_sounds)} sound events in analysis window...")
        
        # Track what each player heard
        player_sounds_heard = {}
        ct_sound_events = []
        audio_timeline = []
        
        for _, sound in window_sounds.iterrows():
            sound_tick = sound['tick']
            sound_player = sound['user_name']
            sound_radius = sound['radius']
            is_footstep = sound['step']
            
            # Get sound source position and team
            source_pos = ticks[(ticks['name'] == sound_player) & (ticks['tick'] == sound_tick)]
            if len(source_pos) == 0:
                continue
            
            source = source_pos.iloc[0]
            source_team = source['team_num']
            source_x, source_y, source_z = source['X'], source['Y'], source['Z']
            
            # Classify zone where sound originated
            source_zone = self.classifier.point_in_zone_A(source_x, source_y, source_z)
            
            # Find all players who could hear this sound
            listeners_at_tick = ticks[ticks['tick'] == sound_tick]
            
            for _, listener in listeners_at_tick.iterrows():
                listener_name = listener['name']
                listener_team = listener['team_num']
                
                # Skip if same player
                if listener_name == sound_player:
                    continue
                
                # Skip if listener not on the team we're analyzing
                if listener_team != team_filter:
                    continue
                
                # Skip if sound maker is on same team as listener (we care about enemy sounds)
                if source_team == listener_team:
                    continue
                
                # Calculate distance
                distance = (
                    (listener['X'] - source_x)**2 + 
                    (listener['Y'] - source_y)**2 + 
                    (listener['Z'] - source_z)**2
                )**0.5
                
                # Check if within hearing range
                if distance <= sound_radius:
                    # Listener could hear this sound!
                    event = {
                        'tick': sound_tick,
                        'time_after_plant': (sound_tick - plant_tick) / 64.0,
                        'listener': listener_name,
                        'sound_maker': sound_player,
                        'sound_maker_team': 'T' if source_team == 2 else 'CT',
                        'distance': distance,
                        'radius': sound_radius,
                        'is_footstep': is_footstep,
                        'source_zone': source_zone,
                        'listener_x': listener['X'],
                        'listener_y': listener['Y'],
                        'listener_z': listener['Z']
                    }
                    
                    # Add to player's heard list
                    if listener_name not in player_sounds_heard:
                        player_sounds_heard[listener_name] = []
                    player_sounds_heard[listener_name].append(event)
                    
                    # If CT sound heard by T, track it
                    if source_team == 3:  # CT
                        ct_sound_events.append(event)
                        audio_timeline.append(event)
        
        # Sort timeline by time
        audio_timeline.sort(key=lambda x: x['tick'])
        
        return {
            'player_sounds_heard': player_sounds_heard,
            'ct_sound_events': ct_sound_events,
            'audio_timeline': audio_timeline
        }
    
    def get_known_ct_zones(self, audio_data: Dict, before_tick: int = None) -> Set[str]:
        """
        Get set of zones where CTs were detected via audio.
        
        Args:
            audio_data: Output from parse_audio_events
            before_tick: Only include sounds before this tick (optional)
        """
        known_zones = set()
        
        for event in audio_data['ct_sound_events']:
            if before_tick and event['tick'] >= before_tick:
                continue
            if event['source_zone']:
                known_zones.add(event['source_zone'])
        
        return known_zones
    
    def get_information_state_at_tick(self, audio_data: Dict, query_tick: int, 
                                      memory_window: int = 320) -> Dict:
        """
        Get information state at a specific tick - what CT zones are known.
        
        Args:
            audio_data: Output from parse_audio_events
            query_tick: Tick to query information state at
            memory_window: How many ticks back to remember sounds (default: 5 seconds = 320 ticks)
        
        Returns dict with:
        - known_ct_zones: Dict[zone_name] -> most recent detection tick
        - recent_sounds: List of sounds heard in memory window
        - ct_players_detected: Dict[player_name] -> most recent zone
        """
        cutoff_tick = query_tick - memory_window
        
        known_ct_zones = {}  # zone -> most recent tick
        ct_players_detected = {}  # player -> (zone, tick)
        recent_sounds = []
        
        for event in audio_data['ct_sound_events']:
            event_tick = event['tick']
            
            # Skip future events
            if event_tick > query_tick:
                continue
            
            # Track recent sounds
            if event_tick >= cutoff_tick:
                recent_sounds.append(event)
            
            # Track zone detections (keep most recent)
            zone = event['source_zone']
            if zone:
                if zone not in known_ct_zones or event_tick > known_ct_zones[zone]:
                    known_ct_zones[zone] = event_tick
            
            # Track player locations (keep most recent)
            player = event['sound_maker']
            if player not in ct_players_detected or event_tick > ct_players_detected[player][1]:
                ct_players_detected[player] = (zone, event_tick)
        
        return {
            'known_ct_zones': known_ct_zones,
            'recent_sounds': recent_sounds,
            'ct_players_detected': ct_players_detected,
            'query_tick': query_tick,
            'memory_window_ticks': memory_window
        }
    
    def build_information_timeline(self, audio_data: Dict, start_tick: int, end_tick: int,
                                   sample_rate: int = 64) -> List[Dict]:
        """
        Build tick-by-tick information state timeline.
        
        Args:
            audio_data: Output from parse_audio_events
            start_tick: Start of timeline
            end_tick: End of timeline
            sample_rate: How often to sample (default: every second = 64 ticks)
        
        Returns list of information states at each sample point
        """
        timeline = []
        
        for tick in range(start_tick, end_tick + 1, sample_rate):
            info_state = self.get_information_state_at_tick(audio_data, tick)
            info_state['tick'] = tick
            info_state['time_offset'] = (tick - start_tick) / 64.0
            timeline.append(info_state)
        
        return timeline
    
    def print_audio_summary(self, audio_data: Dict, plant_tick: int):
        """Print formatted summary of audio events."""
        
        print(f"\n{'='*80}")
        print("AUDIO INFORMATION SUMMARY")
        print(f"{'='*80}\n")
        
        ct_sounds = audio_data['ct_sound_events']
        
        if len(ct_sounds) == 0:
            print("No CT sounds detected by T players")
            return
        
        print(f"CT sounds heard by T players: {len(ct_sounds)} total\n")
        
        # Group by listener
        by_listener = {}
        for event in ct_sounds:
            listener = event['listener']
            if listener not in by_listener:
                by_listener[listener] = []
            by_listener[listener].append(event)
        
        for player, events in sorted(by_listener.items()):
            print(f"{player} heard {len(events)} CT sounds:")
            
            # Sort by time
            events.sort(key=lambda x: x['tick'])
            
            for event in events:
                sound_type = "FOOTSTEP" if event['is_footstep'] else "OTHER"
                print(f"  +{event['time_after_plant']:5.1f}s: {event['sound_maker']:12s} in {event['source_zone']}")
                print(f"          Type: {sound_type}, Distance: {event['distance']:.0f} units (radius: {event['radius']:.0f})")
            print()
        
        # Show CT zones revealed
        known_zones = self.get_known_ct_zones(audio_data)
        print(f"{'='*80}")
        print(f"CT ZONES REVEALED VIA AUDIO: {len(known_zones)} zones")
        print(f"{'='*80}\n")
        
        for zone in sorted(known_zones):
            zone_events = [e for e in ct_sounds if e['source_zone'] == zone]
            first_detection = min(e['time_after_plant'] for e in zone_events)
            print(f"  {zone:30s} (first detected at +{first_detection:.1f}s)")
        
        # Timeline of first detections
        print(f"\n{'='*80}")
        print("AUDIO TIMELINE (First CT detection in each zone)")
        print(f"{'='*80}\n")
        
        detected_zones = set()
        for event in sorted(ct_sounds, key=lambda x: x['tick']):
            zone = event['source_zone']
            if zone not in detected_zones:
                detected_zones.add(zone)
                print(f"  +{event['time_after_plant']:5.1f}s: {event['listener']:12s} hears {event['sound_maker']:12s} in {zone}")
