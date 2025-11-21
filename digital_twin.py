"""
Advanced Digital Twin Engine for Solar Asset Monitoring
Simulates realistic solar farm behavior with multiple sensors and failure modes
"""
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

# Global scenario state
current_scenario = "normal"
asset_states = {}  # Track state per asset

# Failure modes and degradation patterns
FAILURE_MODES = {
    "panel_degradation": {"rate": 0.0001, "impact": "power"},
    "inverter_fault": {"rate": 0.00005, "impact": "efficiency"},
    "battery_aging": {"rate": 0.0002, "impact": "capacity"},
    "thermal_stress": {"rate": 0.00015, "impact": "temperature"},
    "soiling": {"rate": 0.0003, "impact": "power"},
    "shading": {"rate": 0.0001, "impact": "power"},
}

def set_scenario(s: str, asset_id: str = "ASSET-001"):
    """Set failure scenario for an asset"""
    global current_scenario, asset_states
    if s in ("normal", "high_temp", "critical_failure", "degradation", "inverter_fault", "battery_failure"):
        if asset_id not in asset_states:
            asset_states[asset_id] = {}
        asset_states[asset_id]["scenario"] = s
        current_scenario = s
    else:
        current_scenario = "normal"

def initialize_asset(asset_id: str, capacity_kw: float = 500.0):
    """Initialize asset state"""
    if asset_id not in asset_states:
        asset_states[asset_id] = {
            "scenario": "normal",
            "capacity_kw": capacity_kw,
            "age_days": random.randint(0, 365 * 3),
            "degradation_factor": random.uniform(0.95, 1.0),
            "inverter_efficiency": random.uniform(92, 98),
            "battery_capacity": random.uniform(85, 100),
            "panel_health": random.uniform(90, 100),
            "last_maintenance": datetime.now() - timedelta(days=random.randint(30, 180)),
            "failure_risk": random.uniform(0.01, 0.1),
        }

def generate_twin_data(asset_id: str = "ASSET-001", hour_of_day: Optional[int] = None) -> Dict:
    """
    Generate realistic telemetry data with physics-based modeling
    """
    # Initialize asset if needed
    if asset_id not in asset_states:
        initialize_asset(asset_id)
    
    state = asset_states[asset_id]
    scenario = state.get("scenario", current_scenario)
    
    # Time-based solar irradiance (realistic day/night cycle)
    if hour_of_day is None:
        hour_of_day = datetime.now().hour
    
    # Solar irradiance curve (W/m²) - peaks at noon
    base_irradiance = 0
    if 6 <= hour_of_day <= 18:
        # Daytime: bell curve centered at 12:00
        normalized_hour = (hour_of_day - 6) / 12.0
        base_irradiance = 800 * math.sin(math.pi * normalized_hour) ** 2
    irradiance = max(0, base_irradiance + random.uniform(-50, 50))
    
    # Temperature varies with time of day and irradiance
    base_temp = 25 + (irradiance / 30) + random.uniform(-3, 3)
    
    # Power generation (physics-based)
    panel_efficiency = 0.20  # 20% typical efficiency
    area_m2 = state["capacity_kw"] * 1000 / (1000 * panel_efficiency)  # Approximate area
    raw_power = (irradiance * area_m2 * panel_efficiency) / 1000  # kW
    
    # Apply degradation and scenario effects
    degradation = state["degradation_factor"]
    panel_health = state["panel_health"] / 100.0
    
    if scenario == "normal":
        solar_power = raw_power * degradation * panel_health * random.uniform(0.95, 1.05)
        temperature = base_temp + random.uniform(-2, 2)
        battery_level = state["battery_capacity"] + random.uniform(-5, 5)
        
    elif scenario == "high_temp":
        solar_power = raw_power * degradation * panel_health * random.uniform(0.85, 0.95)
        temperature = base_temp + random.uniform(15, 25)  # Overheating
        battery_level = state["battery_capacity"] * 0.7 + random.uniform(-5, 5)
        
    elif scenario == "critical_failure":
        solar_power = raw_power * degradation * 0.3 * random.uniform(0.8, 1.2)
        temperature = base_temp + random.uniform(25, 35)
        battery_level = max(5, state["battery_capacity"] * 0.3 + random.uniform(-3, 3))
        
    elif scenario == "degradation":
        solar_power = raw_power * degradation * 0.75 * random.uniform(0.9, 1.0)
        temperature = base_temp + random.uniform(0, 5)
        battery_level = state["battery_capacity"] * 0.85 + random.uniform(-3, 3)
        
    elif scenario == "inverter_fault":
        solar_power = raw_power * degradation * panel_health * 0.65 * random.uniform(0.9, 1.1)
        temperature = base_temp + random.uniform(5, 10)
        battery_level = state["battery_capacity"] * 0.8 + random.uniform(-3, 3)
        
    elif scenario == "battery_failure":
        solar_power = raw_power * degradation * panel_health * random.uniform(0.9, 1.0)
        temperature = base_temp + random.uniform(0, 3)
        battery_level = max(10, state["battery_capacity"] * 0.4 + random.uniform(-5, 5))
        
    else:
        solar_power = raw_power * degradation * panel_health
        temperature = base_temp
        battery_level = state["battery_capacity"]
    
    # Additional sensor readings
    voltage = 400 + (solar_power / state["capacity_kw"]) * 100 + random.uniform(-10, 10)
    current = (solar_power * 1000) / voltage + random.uniform(-0.5, 0.5)
    
    # Inverter efficiency (degrades with temperature and age)
    inverter_eff = state["inverter_efficiency"]
    if temperature > 50:
        inverter_eff *= (1 - (temperature - 50) * 0.002)
    if scenario == "inverter_fault":
        inverter_eff *= 0.75
    
    # Environmental data
    wind_speed = random.uniform(2, 15)
    humidity = random.uniform(30, 70)
    
    # Calculate power output considering inverter efficiency
    actual_power = solar_power * (inverter_eff / 100.0)
    
    # Update degradation over time (slow)
    if random.random() < 0.01:  # 1% chance per call
        state["degradation_factor"] *= 0.9999
        state["panel_health"] = max(50, state["panel_health"] - 0.01)
    
    return {
        "asset_id": asset_id,
        "timestamp": datetime.now().isoformat(),
        "solar_power": round(actual_power, 2),
        "temperature": round(temperature, 2),
        "battery_level": round(max(0, min(100, battery_level)), 2),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "irradiance": round(irradiance, 2),
        "inverter_efficiency": round(inverter_eff, 2),
        "panel_health": round(state["panel_health"], 2),
        "wind_speed": round(wind_speed, 2),
        "humidity": round(humidity, 2),
        "scenario": scenario,
        "capacity_kw": state["capacity_kw"],
    }

def get_asset_state(asset_id: str) -> Dict:
    """Get current asset state"""
    if asset_id not in asset_states:
        initialize_asset(asset_id)
    return asset_states[asset_id].copy()

def simulate_failure_progression(asset_id: str, days_ahead: int = 30) -> List[Dict]:
    """Simulate future failure progression for predictive analysis"""
    if asset_id not in asset_states:
        initialize_asset(asset_id)
    
    state = asset_states[asset_id]
    progression = []
    current_time = datetime.now()
    
    for day in range(days_ahead):
        for hour in range(0, 24, 2):  # Every 2 hours
            timestamp = current_time + timedelta(days=day, hours=hour)
            data = generate_twin_data(asset_id, hour_of_day=hour)
            data["timestamp"] = timestamp.isoformat()
            
            # Apply progressive degradation
            degradation_rate = state.get("failure_risk", 0.05) * 0.001
            state["degradation_factor"] *= (1 - degradation_rate)
            state["panel_health"] = max(0, state["panel_health"] - degradation_rate * 10)
            
            progression.append(data)
    
    return progression
