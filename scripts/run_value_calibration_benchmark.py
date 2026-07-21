#!/usr/bin/env python3
"""Out-of-sample valuation-calibration benchmark: generate and evaluate.

This script builds and scores a synthetic bundle-pricing benchmark that is
**independent of the PC-build auction scenarios** (``scenarios/pc_build_v1/``).
Its purpose is narrow: calibrate provisional-valuation (PV) parameters
(``epsilon``, ``discount_inferred``, anchor-value usage, etc.) against hidden
ground-truth bundle values, before those parameters are frozen and carried
into auction experiments. See ``docs/parameter_tuning_methodology.md`` for
why this separation matters.

No live LLM/API calls are made by this script. ``generate`` builds a
deterministic synthetic benchmark from a seeded valuation model (reusing
:mod:`auctionlab.instances.structured`'s substitute/complement/saturation
machinery). ``evaluate`` scores a *cached/frozen* PV output file (or, for
testing/demo purposes, a deterministically synthesized fake PV output)
against the benchmark's hidden ground truth -- it never calls a model
itself.

Usage::

    # Generate one benchmark file per domain.
    ./venv/bin/python scripts/run_value_calibration_benchmark.py generate \\
        --domain home_office travel_package --num-goods 6 --num-bidders 4 \\
        --seed 0 --output-dir benchmarks/value_calibration

    # Evaluate a cached PV output file against a generated benchmark.
    ./venv/bin/python scripts/run_value_calibration_benchmark.py evaluate \\
        --benchmark-file benchmarks/value_calibration/value_calibration_benchmark_home_office_6x4_seed0.json \\
        --pv-file my_cached_pv_output.json \\
        --output-dir benchmarks/value_calibration/reports/home_office

    # Or, with no cached PV file yet, exercise the full evaluation pipeline
    # against a deterministic synthetic fake PV (no LLM call of any kind):
    ./venv/bin/python scripts/run_value_calibration_benchmark.py evaluate \\
        --benchmark-file benchmarks/value_calibration/value_calibration_benchmark_home_office_6x4_seed0.json \\
        --synthesize-fake-pv --noise-scale 0.15 --bias-per-size 0.03 --seed 1 \\
        --output-dir benchmarks/value_calibration/reports/home_office_fake
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from auctionlab.instances.structured import (
    BidderPreferenceProfile,
    ComplementGroup,
    SubstituteGroup,
    _comp_if_available,
    _jitter,
    _sub_if_available,
    generate_full_valuations,
)

SCHEMA_VERSION = "value_calibration_benchmark_v1"


# ---------------------------------------------------------------------------
# Domain catalogs (deterministic synthetic valuation model, non-PC-build)
# ---------------------------------------------------------------------------
# Each domain is plain data: a good catalog + item descriptions, and a list
# of bidder archetypes (role, budget range, reference base values, and
# substitute/complement structure). Archetypes are converted into
# `BidderPreferenceProfile`s by `_build_profile`, reusing the same
# jitter/substitute/complement-filtering helpers `instances/structured.py`
# uses for the PC-build domain -- the *data* is independent, but the
# deterministic transformation machinery is shared on purpose.

DOMAIN_CATALOGS: dict[str, dict[str, Any]] = {
    "home_office": {
        "goods": {
            "DESK": "Adjustable-height desk with cable management.",
            "CHAIR": "Ergonomic office chair with lumbar support.",
            "MONITOR": "27-inch 1440p monitor with an adjustable stand.",
            "WEBCAM": "1080p webcam with a built-in privacy shutter.",
            "DESK_LAMP": "LED desk lamp with adjustable color temperature.",
            "DOCKING_STATION": "USB-C docking station for a single-cable laptop setup.",
        },
        "scenario_description": (
            "A home-office equipment auction. Bidders are furnishing or "
            "upgrading a home workspace and have varying needs for comfort, "
            "video-call quality, and desk ergonomics."
        ),
        "bidders": [
            {
                "bidder_id": "remote_worker",
                "role": (
                    "Priya works from home full-time and wants a comfortable, "
                    "well-equipped everyday workspace."
                ),
                "budget_range": (400.0, 900.0),
                "base_values": {
                    "DESK": 220.0, "CHAIR": 260.0, "MONITOR": 180.0,
                    "WEBCAM": 60.0, "DESK_LAMP": 40.0, "DOCKING_STATION": 90.0,
                },
                "complement_groups": [
                    (frozenset({"DESK", "CHAIR", "MONITOR"}), 80.0,
                     "a complete, comfortable everyday workstation"),
                ],
                "core_items": frozenset({"DESK", "CHAIR", "MONITOR"}),
                "secondary_items": frozenset({"DOCKING_STATION", "DESK_LAMP"}),
                "low_interest_items": frozenset({"WEBCAM"}),
            },
            {
                "bidder_id": "video_call_manager",
                "role": (
                    "Jordan runs back-to-back video calls all day and cares "
                    "most about camera and lighting quality on screen."
                ),
                "budget_range": (250.0, 600.0),
                "base_values": {
                    "WEBCAM": 150.0, "DESK_LAMP": 90.0, "MONITOR": 140.0,
                    "DESK": 90.0, "CHAIR": 120.0, "DOCKING_STATION": 70.0,
                },
                "complement_groups": [
                    (frozenset({"WEBCAM", "DESK_LAMP"}), 40.0,
                     "good lighting and a good camera together look far better on camera"),
                ],
                "core_items": frozenset({"WEBCAM", "DESK_LAMP"}),
                "secondary_items": frozenset({"MONITOR", "CHAIR"}),
                "low_interest_items": frozenset({"DESK", "DOCKING_STATION"}),
            },
            {
                "bidder_id": "minimalist_freelancer",
                "role": (
                    "Sam freelances from a small apartment and wants the "
                    "fewest items that still make the desk usable."
                ),
                "budget_range": (150.0, 350.0),
                "base_values": {
                    "DESK": 130.0, "CHAIR": 110.0, "MONITOR": 90.0,
                    "WEBCAM": 30.0, "DESK_LAMP": 35.0, "DOCKING_STATION": 40.0,
                },
                "saturation_start": 3,
                "saturation_penalty": 12.0,
                "core_items": frozenset({"DESK", "CHAIR"}),
                "secondary_items": frozenset({"MONITOR"}),
                "low_interest_items": frozenset({"WEBCAM", "DESK_LAMP", "DOCKING_STATION"}),
            },
            {
                "bidder_id": "ergonomics_focused_manager",
                "role": (
                    "Morgan has chronic back pain and prioritises seating "
                    "and desk ergonomics above all else."
                ),
                "budget_range": (500.0, 1000.0),
                "base_values": {
                    "CHAIR": 420.0, "DESK": 300.0, "MONITOR": 150.0,
                    "DESK_LAMP": 45.0, "WEBCAM": 40.0, "DOCKING_STATION": 60.0,
                },
                "complement_groups": [
                    (frozenset({"CHAIR", "DESK"}), 60.0,
                     "a height-adjustable desk paired with a supportive chair prevents back pain"),
                ],
                "core_items": frozenset({"CHAIR", "DESK"}),
                "secondary_items": frozenset({"MONITOR"}),
                "low_interest_items": frozenset({"WEBCAM", "DESK_LAMP", "DOCKING_STATION"}),
            },
        ],
    },
    "travel_package": {
        "goods": {
            "FLIGHT": "Round-trip economy flight.",
            "HOTEL": "4-night hotel stay in the destination city.",
            "CAR_RENTAL": "Compact car rental for the duration of the trip.",
            "TRAVEL_INSURANCE": "Trip cancellation and medical travel insurance.",
            "GUIDED_TOUR": "A half-day guided city tour with a local guide.",
            "AIRPORT_LOUNGE_PASS": "Single-visit airport lounge access pass.",
        },
        "scenario_description": (
            "A bundled travel-package auction. Bidders are booking a trip "
            "and have different priorities around comfort, flexibility, "
            "and cost."
        ),
        "bidders": [
            {
                "bidder_id": "family_vacationer",
                "role": (
                    "The Alvarez family is planning a relaxed one-week "
                    "vacation and values convenience over flexibility."
                ),
                "budget_range": (1200.0, 2200.0),
                "base_values": {
                    "FLIGHT": 650.0, "HOTEL": 700.0, "CAR_RENTAL": 220.0,
                    "TRAVEL_INSURANCE": 90.0, "GUIDED_TOUR": 140.0,
                    "AIRPORT_LOUNGE_PASS": 50.0,
                },
                "complement_groups": [
                    (frozenset({"FLIGHT", "HOTEL", "CAR_RENTAL"}), 120.0,
                     "a fully-booked trip removes all logistics stress"),
                ],
                "core_items": frozenset({"FLIGHT", "HOTEL", "CAR_RENTAL"}),
                "secondary_items": frozenset({"TRAVEL_INSURANCE", "GUIDED_TOUR"}),
                "low_interest_items": frozenset({"AIRPORT_LOUNGE_PASS"}),
            },
            {
                "bidder_id": "business_traveler",
                "role": (
                    "Dana is on a tight one-night business trip and cares "
                    "most about flight comfort and time efficiency."
                ),
                "budget_range": (600.0, 1400.0),
                "base_values": {
                    "FLIGHT": 500.0, "AIRPORT_LOUNGE_PASS": 180.0,
                    "HOTEL": 280.0, "CAR_RENTAL": 150.0,
                    "TRAVEL_INSURANCE": 40.0, "GUIDED_TOUR": 20.0,
                },
                "complement_groups": [
                    (frozenset({"FLIGHT", "AIRPORT_LOUNGE_PASS"}), 60.0,
                     "a productive, comfortable travel day matters most"),
                ],
                "core_items": frozenset({"FLIGHT", "AIRPORT_LOUNGE_PASS"}),
                "secondary_items": frozenset({"HOTEL", "CAR_RENTAL"}),
                "low_interest_items": frozenset({"TRAVEL_INSURANCE", "GUIDED_TOUR"}),
            },
            {
                "bidder_id": "budget_backpacker",
                "role": (
                    "Riley is backpacking on a strict budget and only wants "
                    "the essentials, skipping anything that feels optional."
                ),
                "budget_range": (300.0, 700.0),
                "base_values": {
                    "FLIGHT": 400.0, "HOTEL": 180.0, "CAR_RENTAL": 60.0,
                    "TRAVEL_INSURANCE": 50.0, "GUIDED_TOUR": 70.0,
                    "AIRPORT_LOUNGE_PASS": 10.0,
                },
                "saturation_start": 3,
                "saturation_penalty": 25.0,
                "core_items": frozenset({"FLIGHT", "HOTEL"}),
                "secondary_items": frozenset({"GUIDED_TOUR", "TRAVEL_INSURANCE"}),
                "low_interest_items": frozenset({"CAR_RENTAL", "AIRPORT_LOUNGE_PASS"}),
            },
            {
                "bidder_id": "luxury_honeymooner",
                "role": (
                    "Chris and Avery are planning a honeymoon and want "
                    "every part of the trip to feel effortless and special."
                ),
                "budget_range": (2500.0, 4500.0),
                "base_values": {
                    "FLIGHT": 900.0, "HOTEL": 1400.0, "CAR_RENTAL": 300.0,
                    "TRAVEL_INSURANCE": 150.0, "GUIDED_TOUR": 350.0,
                    "AIRPORT_LOUNGE_PASS": 200.0,
                },
                "complement_groups": [
                    (frozenset({"HOTEL", "GUIDED_TOUR"}), 150.0,
                     "a premium stay paired with a curated tour feels cohesive"),
                    (frozenset({"FLIGHT", "AIRPORT_LOUNGE_PASS"}), 100.0,
                     "lounge access makes long-haul flights feel like part of the trip"),
                ],
                "core_items": frozenset({"HOTEL", "FLIGHT", "GUIDED_TOUR"}),
                "secondary_items": frozenset({"AIRPORT_LOUNGE_PASS", "CAR_RENTAL"}),
                "low_interest_items": frozenset({"TRAVEL_INSURANCE"}),
            },
        ],
    },
    "camera_video_kit": {
        "goods": {
            "CAMERA_BODY": "Mirrorless camera body with in-body stabilization.",
            "LENS_WIDE": "16-35mm wide-angle zoom lens.",
            "LENS_TELEPHOTO": "70-200mm telephoto zoom lens.",
            "TRIPOD": "Carbon-fiber tripod with a fluid video head.",
            "LIGHTING_KIT": "Two-point LED lighting kit with softboxes.",
            "MEMORY_CARDS": "Set of three high-speed memory cards.",
        },
        "scenario_description": (
            "A camera and video production equipment auction. Bidders are "
            "assembling a kit for different kinds of shooting work."
        ),
        "bidders": [
            {
                "bidder_id": "wildlife_photographer",
                "role": (
                    "Kai shoots wildlife and needs reach and a reliable body "
                    "above all else."
                ),
                "budget_range": (1800.0, 3200.0),
                "base_values": {
                    "CAMERA_BODY": 1400.0, "LENS_TELEPHOTO": 1300.0,
                    "LENS_WIDE": 250.0, "TRIPOD": 300.0,
                    "LIGHTING_KIT": 60.0, "MEMORY_CARDS": 150.0,
                },
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "LENS_TELEPHOTO"}), 200.0,
                     "the body and telephoto lens together are the core wildlife setup"),
                ],
                "core_items": frozenset({"CAMERA_BODY", "LENS_TELEPHOTO"}),
                "secondary_items": frozenset({"TRIPOD", "MEMORY_CARDS"}),
                "low_interest_items": frozenset({"LENS_WIDE", "LIGHTING_KIT"}),
            },
            {
                "bidder_id": "wedding_videographer",
                "role": (
                    "Elena films weddings and needs stable, well-lit footage "
                    "across a wide range of shots in one day."
                ),
                "budget_range": (2200.0, 4000.0),
                "base_values": {
                    "CAMERA_BODY": 1300.0, "TRIPOD": 500.0,
                    "LIGHTING_KIT": 450.0, "LENS_WIDE": 400.0,
                    "LENS_TELEPHOTO": 300.0, "MEMORY_CARDS": 180.0,
                },
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "TRIPOD", "LIGHTING_KIT"}), 250.0,
                     "stable, well-lit shots are non-negotiable for wedding delivery"),
                ],
                "core_items": frozenset({"CAMERA_BODY", "TRIPOD", "LIGHTING_KIT"}),
                "secondary_items": frozenset({"LENS_WIDE", "MEMORY_CARDS"}),
                "low_interest_items": frozenset({"LENS_TELEPHOTO"}),
            },
            {
                "bidder_id": "hobbyist_vlogger",
                "role": (
                    "Theo vlogs as a hobby on evenings and weekends and "
                    "keeps the kit small and affordable."
                ),
                "budget_range": (400.0, 900.0),
                "base_values": {
                    "CAMERA_BODY": 500.0, "LENS_WIDE": 200.0,
                    "TRIPOD": 120.0, "LIGHTING_KIT": 150.0,
                    "LENS_TELEPHOTO": 60.0, "MEMORY_CARDS": 70.0,
                },
                "saturation_start": 3,
                "saturation_penalty": 30.0,
                "core_items": frozenset({"CAMERA_BODY", "LENS_WIDE"}),
                "secondary_items": frozenset({"LIGHTING_KIT", "TRIPOD"}),
                "low_interest_items": frozenset({"LENS_TELEPHOTO", "MEMORY_CARDS"}),
            },
            {
                "bidder_id": "studio_portrait_photographer",
                "role": (
                    "Nina shoots portraits in a fixed studio and prioritises "
                    "lighting control over lens reach or portability."
                ),
                "budget_range": (1200.0, 2400.0),
                "base_values": {
                    "LIGHTING_KIT": 700.0, "CAMERA_BODY": 900.0,
                    "LENS_WIDE": 150.0, "TRIPOD": 200.0,
                    "LENS_TELEPHOTO": 100.0, "MEMORY_CARDS": 90.0,
                },
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "LIGHTING_KIT"}), 180.0,
                     "controlled studio lighting paired with the body defines the look"),
                ],
                "core_items": frozenset({"LIGHTING_KIT", "CAMERA_BODY"}),
                "secondary_items": frozenset({"TRIPOD"}),
                "low_interest_items": frozenset({"LENS_WIDE", "LENS_TELEPHOTO", "MEMORY_CARDS"}),
            },
        ],
    },
    "kitchen_appliance_bundle": {
        "goods": {
            "STAND_MIXER": "Stand mixer with dough hook and whisk attachments.",
            "BLENDER": "High-power countertop blender.",
            "AIR_FRYER": "6-quart digital air fryer.",
            "COFFEE_MACHINE": "Espresso machine with a built-in grinder.",
            "SOUS_VIDE": "Immersion sous vide precision cooker.",
            "FOOD_PROCESSOR": "Multi-blade food processor with slicing discs.",
        },
        "scenario_description": (
            "A kitchen appliance bundle auction. Bidders are outfitting a "
            "kitchen for different styles of home cooking."
        ),
        "bidders": [
            {
                "bidder_id": "home_baker",
                "role": (
                    "Grace bakes bread and pastries most weekends and wants "
                    "reliable mixing and prep equipment."
                ),
                "budget_range": (350.0, 800.0),
                "base_values": {
                    "STAND_MIXER": 380.0, "FOOD_PROCESSOR": 180.0,
                    "BLENDER": 90.0, "AIR_FRYER": 60.0,
                    "COFFEE_MACHINE": 70.0, "SOUS_VIDE": 40.0,
                },
                "complement_groups": [
                    (frozenset({"STAND_MIXER", "FOOD_PROCESSOR"}), 50.0,
                     "mixing and prep together cover nearly all baking tasks"),
                ],
                "core_items": frozenset({"STAND_MIXER", "FOOD_PROCESSOR"}),
                "secondary_items": frozenset({"BLENDER"}),
                "low_interest_items": frozenset({"AIR_FRYER", "COFFEE_MACHINE", "SOUS_VIDE"}),
            },
            {
                "bidder_id": "health_conscious_cook",
                "role": (
                    "Marcus cooks high-protein meals daily and wants precise, "
                    "repeatable results with minimal added fat."
                ),
                "budget_range": (300.0, 700.0),
                "base_values": {
                    "SOUS_VIDE": 320.0, "AIR_FRYER": 210.0,
                    "BLENDER": 150.0, "FOOD_PROCESSOR": 100.0,
                    "STAND_MIXER": 40.0, "COFFEE_MACHINE": 60.0,
                },
                "complement_groups": [
                    (frozenset({"SOUS_VIDE", "AIR_FRYER"}), 40.0,
                     "sous-vide followed by an air-fryer sear covers most weeknight meals"),
                ],
                "core_items": frozenset({"SOUS_VIDE", "AIR_FRYER"}),
                "secondary_items": frozenset({"BLENDER", "FOOD_PROCESSOR"}),
                "low_interest_items": frozenset({"STAND_MIXER", "COFFEE_MACHINE"}),
            },
            {
                "bidder_id": "busy_parent",
                "role": (
                    "Aisha is cooking for a family on a tight weeknight "
                    "schedule and values speed and easy cleanup."
                ),
                "budget_range": (200.0, 500.0),
                "base_values": {
                    "AIR_FRYER": 220.0, "BLENDER": 130.0,
                    "FOOD_PROCESSOR": 110.0, "STAND_MIXER": 60.0,
                    "COFFEE_MACHINE": 90.0, "SOUS_VIDE": 40.0,
                },
                "saturation_start": 3,
                "saturation_penalty": 18.0,
                "core_items": frozenset({"AIR_FRYER", "BLENDER"}),
                "secondary_items": frozenset({"FOOD_PROCESSOR", "COFFEE_MACHINE"}),
                "low_interest_items": frozenset({"STAND_MIXER", "SOUS_VIDE"}),
            },
            {
                "bidder_id": "coffee_enthusiast",
                "role": (
                    "Leo is a self-described coffee obsessive who cares far "
                    "more about the coffee machine than anything else in "
                    "the kitchen."
                ),
                "budget_range": (250.0, 650.0),
                "base_values": {
                    "COFFEE_MACHINE": 480.0, "BLENDER": 80.0,
                    "AIR_FRYER": 50.0, "FOOD_PROCESSOR": 40.0,
                    "STAND_MIXER": 30.0, "SOUS_VIDE": 30.0,
                },
                "core_items": frozenset({"COFFEE_MACHINE"}),
                "secondary_items": frozenset({"BLENDER"}),
                "low_interest_items": frozenset(
                    {"AIR_FRYER", "FOOD_PROCESSOR", "STAND_MIXER", "SOUS_VIDE"}
                ),
            },
        ],
    },
    "gaming_peripherals": {
        "goods": {
            "MECHANICAL_KEYBOARD": "Hot-swappable mechanical keyboard.",
            "GAMING_MOUSE": "Lightweight wireless gaming mouse.",
            "HEADSET": "Wireless gaming headset with a detachable microphone.",
            "MONITOR_144HZ": "27-inch 144Hz gaming monitor.",
            "CONTROLLER": "Wireless controller with remappable back buttons.",
            "MOUSE_PAD": "Extended desk-size mouse pad.",
        },
        "scenario_description": (
            "A gaming peripherals auction. Bidders are equipping different "
            "gaming setups, from competitive PC play to relaxed console play."
        ),
        "bidders": [
            {
                "bidder_id": "competitive_fps_player",
                "role": (
                    "Zoe plays competitive first-person shooters and needs "
                    "the lowest possible input latency and highest refresh rate."
                ),
                "budget_range": (500.0, 1000.0),
                "base_values": {
                    "MONITOR_144HZ": 420.0, "GAMING_MOUSE": 180.0,
                    "MECHANICAL_KEYBOARD": 160.0, "MOUSE_PAD": 40.0,
                    "HEADSET": 120.0, "CONTROLLER": 20.0,
                },
                "complement_groups": [
                    (frozenset({"MONITOR_144HZ", "GAMING_MOUSE", "MECHANICAL_KEYBOARD"}), 70.0,
                     "a matched high-refresh monitor and low-latency mouse/keyboard "
                     "combo is the whole point of a competitive setup"),
                ],
                "core_items": frozenset(
                    {"MONITOR_144HZ", "GAMING_MOUSE", "MECHANICAL_KEYBOARD"}
                ),
                "secondary_items": frozenset({"HEADSET", "MOUSE_PAD"}),
                "low_interest_items": frozenset({"CONTROLLER"}),
            },
            {
                "bidder_id": "controller_console_player",
                "role": (
                    "Marcus plays mostly on console with a controller and "
                    "cares about sound and a comfortable controller."
                ),
                "budget_range": (150.0, 400.0),
                "base_values": {
                    "CONTROLLER": 130.0, "HEADSET": 150.0,
                    "MONITOR_144HZ": 90.0, "GAMING_MOUSE": 10.0,
                    "MECHANICAL_KEYBOARD": 10.0, "MOUSE_PAD": 15.0,
                },
                "complement_groups": [
                    (frozenset({"CONTROLLER", "HEADSET"}), 30.0,
                     "a comfortable controller and a good headset are used together every session"),
                ],
                "core_items": frozenset({"CONTROLLER", "HEADSET"}),
                "secondary_items": frozenset({"MONITOR_144HZ"}),
                "low_interest_items": frozenset(
                    {"GAMING_MOUSE", "MECHANICAL_KEYBOARD", "MOUSE_PAD"}
                ),
            },
            {
                "bidder_id": "streamer_setup_builder",
                "role": (
                    "Priya streams her gameplay and wants everything to "
                    "look and sound good on camera as much as perform well."
                ),
                "budget_range": (600.0, 1200.0),
                "base_values": {
                    "HEADSET": 220.0, "MECHANICAL_KEYBOARD": 210.0,
                    "MONITOR_144HZ": 300.0, "GAMING_MOUSE": 150.0,
                    "MOUSE_PAD": 60.0, "CONTROLLER": 60.0,
                },
                "complement_groups": [
                    (frozenset({"MECHANICAL_KEYBOARD", "MOUSE_PAD"}), 25.0,
                     "a matched keyboard and deskmat look better on stream"),
                ],
                "core_items": frozenset(
                    {"HEADSET", "MECHANICAL_KEYBOARD", "MONITOR_144HZ"}
                ),
                "secondary_items": frozenset({"GAMING_MOUSE", "MOUSE_PAD"}),
                "low_interest_items": frozenset({"CONTROLLER"}),
            },
            {
                "bidder_id": "casual_gamer",
                "role": (
                    "Sam plays casually a few evenings a week and doesn't "
                    "want to overspend on any single item."
                ),
                "budget_range": (100.0, 300.0),
                "base_values": {
                    "GAMING_MOUSE": 60.0, "MECHANICAL_KEYBOARD": 70.0,
                    "HEADSET": 80.0, "MONITOR_144HZ": 90.0,
                    "CONTROLLER": 50.0, "MOUSE_PAD": 15.0,
                },
                "saturation_start": 3,
                "saturation_penalty": 10.0,
                "core_items": frozenset({"HEADSET", "MONITOR_144HZ"}),
                "secondary_items": frozenset({"GAMING_MOUSE", "MECHANICAL_KEYBOARD"}),
                "low_interest_items": frozenset({"CONTROLLER", "MOUSE_PAD"}),
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Profile construction + NL seed rendering (domain-agnostic, no PC_GOOD_CATALOG
# dependency -- deliberately separate from instances/structured.py's
# render_budget_calibrated_seed, which orders items via the PC-specific
# PC_GOOD_CATALOG and would silently omit every item for a new domain).
# ---------------------------------------------------------------------------

def _build_profile(archetype: dict[str, Any], items: set[str], rng: random.Random) -> BidderPreferenceProfile:
    base_values = {
        item: _jitter(value, rng)
        for item, value in archetype["base_values"].items()
        if item in items
    }
    substitute_groups: list[SubstituteGroup] = []
    for group_items, backup_factor, description in archetype.get("substitute_groups", []):
        substitute_groups += _sub_if_available(items, group_items, backup_factor, description)
    complement_groups: list[ComplementGroup] = []
    for group_items, bonus, description in archetype.get("complement_groups", []):
        complement_groups += _comp_if_available(items, group_items, _jitter(bonus, rng), description)

    return BidderPreferenceProfile(
        bidder_id=archetype["bidder_id"],
        role=archetype["role"],
        budget_range=archetype["budget_range"],
        base_values=base_values,
        substitute_groups=substitute_groups,
        complement_groups=complement_groups,
        budget_cap=archetype.get("budget_cap"),
        saturation_start=archetype.get("saturation_start"),
        saturation_penalty=archetype.get("saturation_penalty", 0.0),
        notes=archetype.get("notes", ""),
        core_items=archetype.get("core_items", frozenset()) & items,
        secondary_items=archetype.get("secondary_items", frozenset()) & items,
        low_interest_items=archetype.get("low_interest_items", frozenset()) & items,
    )


def _price_range(value: float) -> str:
    lo = max(5, round(value * 0.85 / 5) * 5)
    hi = round(value * 1.15 / 5) * 5
    return f"${lo:,.0f}-${hi:,.0f}"


def _render_person_seed(profile: BidderPreferenceProfile, items: list[str]) -> str:
    """Domain-agnostic NL seed, ordered by ``items`` rather than a hard-coded
    catalog (unlike ``instances.structured.render_budget_calibrated_seed``,
    which orders complement/substitute mentions via ``PC_GOOD_CATALOG`` and
    would silently omit every item for a non-PC-build domain)."""
    parts: list[str] = [profile.role]
    lo, hi = profile.budget_range
    parts.append(f"Overall budget for this auction is roughly ${lo:,.0f}-${hi:,.0f}.")

    core = sorted(profile.core_items & set(profile.base_values))
    if core:
        item_strs = [
            f"{item} (worth roughly {_price_range(profile.base_values[item])} to them)"
            for item in core
        ]
        parts.append("Highest-priority items: " + "; ".join(item_strs) + ".")

    for cg in profile.complement_groups:
        available = [i for i in items if i in cg.items and i in profile.base_values]
        if len(available) >= 2:
            desc = cg.description or "owning the set together is worth more than the items separately"
            parts.append(f"{', '.join(available)} work together: {desc}.")

    for sg in profile.substitute_groups:
        available = [i for i in items if i in sg.items and i in profile.base_values]
        if len(available) >= 2:
            desc = sg.description or "owning more than one adds limited extra value"
            parts.append(f"Regarding {', '.join(available)}: {desc}.")

    secondary = sorted(profile.secondary_items & set(profile.base_values))
    if secondary:
        parts.append("Secondary interest in: " + ", ".join(secondary) + ".")

    low = sorted(profile.low_interest_items & set(profile.base_values))
    if low:
        parts.append("Limited interest in: " + ", ".join(low) + ".")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Candidate bundle generation (deterministic, priority-ordered)
# ---------------------------------------------------------------------------

def bundle_key(bundle: frozenset[str]) -> str:
    """Stable bundle serialization: ``+``-joined sorted item ids.

    Matches the convention already used by
    ``scenarios/pc_build_v1/diagnostics/pv_vs_gt_comparison.csv``.
    """
    return "+".join(sorted(bundle))


def parse_bundle_key(key: str) -> frozenset[str]:
    return frozenset(key.split("+")) if key else frozenset()


def generate_candidate_bundles(
    items: list[str],
    *,
    max_candidate_bundle_size: int = 2,
) -> list[frozenset[str]]:
    """Deterministic candidate bundle set: small bundles plus large ones.

    Includes every bundle of size ``1..max_candidate_bundle_size`` (the
    bundles a realistic PV call would actually be asked to price), plus the
    grand bundle and every "all-but-one" bundle, specifically so
    large-bundle overvaluation bias is measurable even when
    ``max_candidate_bundle_size`` is small.
    """
    n = len(items)
    seen: set[frozenset[str]] = set()
    bundles: list[frozenset[str]] = []

    def _add(bundle: frozenset[str]) -> None:
        if bundle and bundle not in seen:
            seen.add(bundle)
            bundles.append(bundle)

    for size in range(1, min(max_candidate_bundle_size, n) + 1):
        for combo in itertools.combinations(items, size):
            _add(frozenset(combo))

    if n >= 2:
        for item in items:
            _add(frozenset(items) - {item})
        _add(frozenset(items))

    return bundles


# ---------------------------------------------------------------------------
# generate mode
# ---------------------------------------------------------------------------

def build_benchmark(
    domain: str,
    *,
    num_goods: int,
    num_bidders: int,
    seed: int,
    max_candidate_bundle_size: int = 2,
) -> dict[str, Any]:
    if domain not in DOMAIN_CATALOGS:
        raise ValueError(
            f"unknown domain {domain!r}; available: {sorted(DOMAIN_CATALOGS)}"
        )
    catalog = DOMAIN_CATALOGS[domain]
    all_goods = list(catalog["goods"])
    all_archetypes = catalog["bidders"]

    if not (1 <= num_goods <= len(all_goods)):
        raise ValueError(
            f"num_goods must be between 1 and {len(all_goods)} for domain "
            f"{domain!r}, got {num_goods}"
        )
    if not (1 <= num_bidders <= len(all_archetypes)):
        raise ValueError(
            f"num_bidders must be between 1 and {len(all_archetypes)} for "
            f"domain {domain!r}, got {num_bidders}"
        )

    items = all_goods[:num_goods]
    items_set = set(items)
    archetypes = all_archetypes[:num_bidders]

    rng = random.Random(seed)
    profiles = [_build_profile(archetype, items_set, rng) for archetype in archetypes]

    ground_truth_full = generate_full_valuations(items, profiles)
    candidate_bundles = generate_candidate_bundles(
        items, max_candidate_bundle_size=max_candidate_bundle_size
    )
    candidate_keys = [bundle_key(b) for b in candidate_bundles]

    ground_truth_valuations = {
        profile.bidder_id: {
            bundle_key(bundle): ground_truth_full[profile.bidder_id][bundle]
            for bundle in candidate_bundles
        }
        for profile in profiles
    }
    person_seeds = {
        profile.bidder_id: _render_person_seed(profile, items) for profile in profiles
    }
    bidder_profiles_meta = {
        profile.bidder_id: {
            "role": profile.role,
            "budget_range": list(profile.budget_range),
            "core_items": sorted(profile.core_items),
            "secondary_items": sorted(profile.secondary_items),
            "low_interest_items": sorted(profile.low_interest_items),
            "substitute_groups": [
                {"items": sorted(sg.items), "backup_factor": sg.backup_factor}
                for sg in profile.substitute_groups
            ],
            "complement_groups": [
                {"items": sorted(cg.items), "bonus": cg.bonus}
                for cg in profile.complement_groups
            ],
        }
        for profile in profiles
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "seed": seed,
        "num_goods": num_goods,
        "num_bidders": num_bidders,
        "max_candidate_bundle_size": max_candidate_bundle_size,
        "scenario_description": catalog["scenario_description"],
        "goods": [{"id": item, "description": catalog["goods"][item]} for item in items],
        "item_descriptions": {item: catalog["goods"][item] for item in items},
        "bidder_ids": [profile.bidder_id for profile in profiles],
        "bidder_profiles": bidder_profiles_meta,
        "person_seeds": person_seeds,
        "candidate_bundles": candidate_keys,
        "ground_truth_valuations": ground_truth_valuations,
        "valuation_model": "structured_substitutes_complements",
    }


def benchmark_file_name(benchmark: dict[str, Any]) -> str:
    return (
        f"value_calibration_benchmark_{benchmark['domain']}_"
        f"{benchmark['num_goods']}x{benchmark['num_bidders']}_"
        f"seed{benchmark['seed']}.json"
    )


def write_benchmark(benchmark: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / benchmark_file_name(benchmark)
    path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Fake/synthetic PV generation (deterministic, NOT a live LLM call)
# ---------------------------------------------------------------------------

def synthesize_fake_pv(
    benchmark: dict[str, Any],
    *,
    noise_scale: float = 0.15,
    bias_per_size: float = 0.0,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Deterministically perturb ground truth to fake a PV output.

    Purely for exercising the evaluation pipeline without any cached PV file
    or live model call. ``bias_per_size`` > 0 injects a size-proportional
    upward bias, useful for demonstrating/testing the large-bundle
    overvaluation-bias diagnostic; ``noise_scale`` adds symmetric
    multiplicative noise.
    """
    rng = random.Random(seed)
    gt = benchmark["ground_truth_valuations"]
    fake: dict[str, dict[str, float]] = {}
    for bidder_id, bundle_values in gt.items():
        fake[bidder_id] = {}
        for key, value in bundle_values.items():
            size = len(parse_bundle_key(key))
            biased = value * (1.0 + bias_per_size * size)
            noisy = biased * (1.0 + rng.uniform(-noise_scale, noise_scale))
            fake[bidder_id][key] = max(0.0, noisy)
    return fake


def write_pv_cache(
    pv_by_bidder: dict[str, dict[str, float]],
    path: Path,
    *,
    benchmark_name: str,
    provider: str = "synthetic",
    model: str | None = None,
    pv_max_tokens: int | None = None,
) -> None:
    """Write a cached/frozen PV output file in this script's own PV-cache
    schema (no such convention existed elsewhere in the codebase to reuse)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "pv_cache_v1",
        "benchmark_name": benchmark_name,
        "provider": provider,
        "model": model,
        "pv_max_tokens": pv_max_tokens,
        "pv_by_bidder": pv_by_bidder,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# evaluate mode: diagnostics
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Average ranks (1-indexed), ties sharing the mean rank of their span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / (var_x * var_y) ** 0.5


def topk_recall(
    bundle_keys: list[str],
    gt: dict[str, float],
    pv: dict[str, float],
    k: int,
) -> float | None:
    if k <= 0 or len(bundle_keys) < k:
        return None
    true_top = set(sorted(bundle_keys, key=lambda b: (-gt[b], b))[:k])
    proxy_top = set(sorted(bundle_keys, key=lambda b: (-pv.get(b, 0.0), b))[:k])
    return len(true_top & proxy_top) / k


def build_bundle_level_rows(
    benchmark: dict[str, Any],
    pv_by_bidder: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    domain = benchmark["domain"]
    rows: list[dict[str, Any]] = []
    for bidder_id, gt_values in benchmark["ground_truth_valuations"].items():
        pv_values = pv_by_bidder.get(bidder_id, {})
        for key, gt_value in gt_values.items():
            pv_value = pv_values.get(key, 0.0)
            signed_error = pv_value - gt_value
            abs_error = abs(signed_error)
            pct_error = (signed_error / gt_value) if gt_value > 0 else ""
            rows.append({
                "domain": domain,
                "bidder_id": bidder_id,
                "bundle": key,
                "bundle_size": len(parse_bundle_key(key)),
                "gt_value": gt_value,
                "pv_value": pv_value,
                "signed_error": signed_error,
                "abs_error": abs_error,
                "pct_error": pct_error,
            })
    return rows


def _aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared error-metric aggregation over a list of bundle-level rows."""
    n = len(rows)
    if n == 0:
        return {
            "n_bundles": 0,
            "mean_signed_error": "",
            "mean_abs_error": "",
            "median_abs_error": "",
            "mape": "",
            "mean_pv": "",
            "mean_gt": "",
            "mean_ratio": "",
            "median_pv": "",
            "median_gt": "",
            "median_ratio": "",
        }

    signed_errors = [r["signed_error"] for r in rows]
    abs_errors = [r["abs_error"] for r in rows]
    pv_values = [r["pv_value"] for r in rows]
    gt_values = [r["gt_value"] for r in rows]
    pct_errors = [abs(r["pct_error"]) for r in rows if r["pct_error"] != ""]

    mean_pv = statistics.mean(pv_values)
    mean_gt = statistics.mean(gt_values)
    median_pv = statistics.median(pv_values)
    median_gt = statistics.median(gt_values)

    return {
        "n_bundles": n,
        "mean_signed_error": statistics.mean(signed_errors),
        "mean_abs_error": statistics.mean(abs_errors),
        "median_abs_error": statistics.median(abs_errors),
        "mape": statistics.mean(pct_errors) if pct_errors else "",
        "mean_pv": mean_pv,
        "mean_gt": mean_gt,
        "mean_ratio": (mean_pv / mean_gt) if mean_gt > 0 else "",
        "median_pv": median_pv,
        "median_gt": median_gt,
        "median_ratio": (median_pv / median_gt) if median_gt > 0 else "",
    }


def build_summary_rows(
    benchmark: dict[str, Any],
    pv_by_bidder: dict[str, dict[str, float]],
    bundle_rows: list[dict[str, Any]],
    *,
    large_bundle_size_threshold: int = 4,
    topk_values: tuple[int, ...] = (3, 5),
) -> list[dict[str, Any]]:
    domain = benchmark["domain"]
    rows: list[dict[str, Any]] = []

    for bidder_id, gt_values in benchmark["ground_truth_valuations"].items():
        pv_values = pv_by_bidder.get(bidder_id, {})
        bidder_rows = [r for r in bundle_rows if r["bidder_id"] == bidder_id]
        stats = _aggregate_stats(bidder_rows)

        bundle_keys = list(gt_values)
        rank_corr = spearman_rank_correlation(
            [gt_values[k] for k in bundle_keys],
            [pv_values.get(k, 0.0) for k in bundle_keys],
        )

        large_rows = [r for r in bidder_rows if r["bundle_size"] >= large_bundle_size_threshold]
        small_rows = [r for r in bidder_rows if r["bundle_size"] < large_bundle_size_threshold]
        large_mean = statistics.mean([r["signed_error"] for r in large_rows]) if large_rows else ""
        small_mean = statistics.mean([r["signed_error"] for r in small_rows]) if small_rows else ""
        bias = (large_mean - small_mean) if (large_rows and small_rows) else ""

        row: dict[str, Any] = {
            "domain": domain,
            "bidder_id": bidder_id,
            **stats,
            "rank_correlation": rank_corr if rank_corr is not None else "",
            "large_bundle_mean_signed_error": large_mean,
            "small_bundle_mean_signed_error": small_mean,
            "large_bundle_overvaluation_bias": bias,
        }
        for k in topk_values:
            recall = topk_recall(bundle_keys, gt_values, pv_values, k)
            row[f"topk_recall_k{k}"] = recall if recall is not None else ""
        rows.append(row)

    # Domain-level aggregate ("ALL" bidders).
    domain_rows = [r for r in bundle_rows if r["domain"] == domain]
    domain_stats = _aggregate_stats(domain_rows)
    large_rows = [r for r in domain_rows if r["bundle_size"] >= large_bundle_size_threshold]
    small_rows = [r for r in domain_rows if r["bundle_size"] < large_bundle_size_threshold]
    large_mean = statistics.mean([r["signed_error"] for r in large_rows]) if large_rows else ""
    small_mean = statistics.mean([r["signed_error"] for r in small_rows]) if small_rows else ""
    bias = (large_mean - small_mean) if (large_rows and small_rows) else ""
    domain_row: dict[str, Any] = {
        "domain": domain,
        "bidder_id": "ALL",
        **domain_stats,
        "rank_correlation": "",
        "large_bundle_mean_signed_error": large_mean,
        "small_bundle_mean_signed_error": small_mean,
        "large_bundle_overvaluation_bias": bias,
    }
    for k in topk_values:
        domain_row[f"topk_recall_k{k}"] = ""
    rows.append(domain_row)

    return rows


def build_by_bundle_size_rows(
    bundle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domains = sorted({r["domain"] for r in bundle_rows})

    for domain in domains:
        sizes = sorted({r["bundle_size"] for r in bundle_rows if r["domain"] == domain})
        for size in sizes:
            size_rows = [
                r for r in bundle_rows if r["domain"] == domain and r["bundle_size"] == size
            ]
            rows.append({"domain": domain, "bundle_size": size, **_aggregate_stats(size_rows)})

    all_sizes = sorted({r["bundle_size"] for r in bundle_rows})
    for size in all_sizes:
        size_rows = [r for r in bundle_rows if r["bundle_size"] == size]
        rows.append({"domain": "ALL", "bundle_size": size, **_aggregate_stats(size_rows)})

    return rows


def build_report_markdown(
    benchmarks: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    by_size_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Value-calibration benchmark report",
        "",
        "This report scores a provisional-valuation (PV) output against a "
        "**synthetic, out-of-sample bundle-pricing benchmark** -- "
        "independent of the PC-build auction scenarios. It exists to "
        "calibrate valuation-scale parameters (epsilon, discount_inferred, "
        "anchor-value usage) before they are frozen and carried into "
        "auction experiments. See `docs/parameter_tuning_methodology.md`.",
        "",
        "## Benchmarks evaluated",
        "",
    ]
    for b in benchmarks:
        lines.append(
            f"- `{b['domain']}` ({b['num_goods']}x{b['num_bidders']}, "
            f"seed={b['seed']}, {len(b['candidate_bundles'])} candidate bundles)"
        )

    lines += ["", "## Summary by bidder", "", "| domain | bidder | n | mean_signed_error | mean_abs_error | mape | mean_ratio | rank_corr | large_bundle_bias |", "|---|---|---|---|---|---|---|---|---|"]
    for row in summary_rows:
        def _fmt(v: Any) -> str:
            return f"{v:.3f}" if isinstance(v, float) else ("" if v == "" else str(v))
        lines.append(
            f"| {row['domain']} | {row['bidder_id']} | {row['n_bundles']} | "
            f"{_fmt(row['mean_signed_error'])} | {_fmt(row['mean_abs_error'])} | "
            f"{_fmt(row['mape'])} | {_fmt(row['mean_ratio'])} | "
            f"{_fmt(row['rank_correlation'])} | "
            f"{_fmt(row['large_bundle_overvaluation_bias'])} |"
        )

    lines += ["", "## Error by bundle size", "", "| domain | size | n | mean_signed_error | mean_abs_error | mape |", "|---|---|---|---|---|---|"]
    for row in by_size_rows:
        def _fmt2(v: Any) -> str:
            return f"{v:.3f}" if isinstance(v, float) else ("" if v == "" else str(v))
        lines.append(
            f"| {row['domain']} | {row['bundle_size']} | {row['n_bundles']} | "
            f"{_fmt2(row['mean_signed_error'])} | {_fmt2(row['mean_abs_error'])} | "
            f"{_fmt2(row['mape'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Config schema (mirrors configs/value_calibration_example.json)
# ---------------------------------------------------------------------------

class ValueCalibrationBenchmarkConfig(BaseModel):
    domains: list[str] = Field(default_factory=lambda: ["home_office"])
    num_goods: int = 6
    num_bidders: int = 4
    seed: int = 0
    max_candidate_bundle_size: int = 2
    large_bundle_size_threshold: int = 4
    output_dir: str = "benchmarks/value_calibration"


def load_config(path: str | Path) -> ValueCalibrationBenchmarkConfig:
    return ValueCalibrationBenchmarkConfig.model_validate(
        json.loads(Path(path).read_text())
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gen = subparsers.add_parser("generate", help="Create deterministic benchmark JSON file(s).")
    gen.add_argument("--config", type=str, default=None, help="Optional JSON config file (see configs/value_calibration_example.json).")
    gen.add_argument(
        "--domains", nargs="+", default=None,
        choices=list(DOMAIN_CATALOGS) + ["all"],
        help=(
            "One or more domain names, or 'all' for every registered domain "
            "(default: from --config, or 'home_office'). Use '--config "
            "configs/value_calibration_example.json' to reproducibly generate "
            "the full five-domain benchmark without listing each name."
        ),
    )
    gen.add_argument("--num-goods", type=int, default=None)
    gen.add_argument("--num-bidders", type=int, default=None)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--max-candidate-bundle-size", type=int, default=None)
    gen.add_argument("--output-dir", type=str, default=None)

    ev = subparsers.add_parser("evaluate", help="Score a cached/frozen PV output (or synthetic fake PV) against a benchmark's ground truth.")
    ev.add_argument("--benchmark-file", nargs="+", required=True, help="One or more generated benchmark JSON files.")
    ev.add_argument("--pv-file", nargs="+", default=None, help="One cached PV JSON file per --benchmark-file, same order.")
    ev.add_argument("--synthesize-fake-pv", action="store_true", help="Use a deterministic synthetic fake PV instead of --pv-file (no LLM call). For testing/demoing the evaluation pipeline.")
    ev.add_argument("--noise-scale", type=float, default=0.15, help="--synthesize-fake-pv: multiplicative noise scale.")
    ev.add_argument("--bias-per-size", type=float, default=0.0, help="--synthesize-fake-pv: per-bundle-size multiplicative bias (>0 simulates large-bundle overvaluation).")
    ev.add_argument("--seed", type=int, default=0, help="--synthesize-fake-pv: noise RNG seed.")
    ev.add_argument("--large-bundle-size-threshold", type=int, default=4)
    ev.add_argument("--output-dir", type=str, required=True)

    return parser.parse_args(argv)


def run_generate(args: argparse.Namespace) -> list[Path]:
    config = load_config(args.config) if args.config else ValueCalibrationBenchmarkConfig()
    domains = args.domains or config.domains
    if domains == ["all"]:
        domains = sorted(DOMAIN_CATALOGS)
    num_goods = args.num_goods if args.num_goods is not None else config.num_goods
    num_bidders = args.num_bidders if args.num_bidders is not None else config.num_bidders
    seed = args.seed if args.seed is not None else config.seed
    max_candidate_bundle_size = (
        args.max_candidate_bundle_size
        if args.max_candidate_bundle_size is not None
        else config.max_candidate_bundle_size
    )
    output_dir = Path(args.output_dir if args.output_dir is not None else config.output_dir)

    written: list[Path] = []
    for domain in domains:
        benchmark = build_benchmark(
            domain,
            num_goods=num_goods,
            num_bidders=num_bidders,
            seed=seed,
            max_candidate_bundle_size=max_candidate_bundle_size,
        )
        path = write_benchmark(benchmark, output_dir)
        written.append(path)
        print(f"Wrote {path} ({len(benchmark['candidate_bundles'])} candidate bundles, {len(benchmark['bidder_ids'])} bidders)")
    return written


def run_evaluate(args: argparse.Namespace) -> None:
    if not args.synthesize_fake_pv and not args.pv_file:
        raise SystemExit("evaluate requires --pv-file (one per --benchmark-file) or --synthesize-fake-pv")
    if args.pv_file and len(args.pv_file) != len(args.benchmark_file):
        raise SystemExit("--pv-file must be given once per --benchmark-file, in the same order")

    output_dir = Path(args.output_dir)
    benchmarks: list[dict[str, Any]] = []
    all_bundle_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []

    for idx, benchmark_path in enumerate(args.benchmark_file):
        benchmark = json.loads(Path(benchmark_path).read_text())
        benchmarks.append(benchmark)

        if args.synthesize_fake_pv:
            pv_by_bidder = synthesize_fake_pv(
                benchmark,
                noise_scale=args.noise_scale,
                bias_per_size=args.bias_per_size,
                seed=args.seed,
            )
        else:
            pv_payload = json.loads(Path(args.pv_file[idx]).read_text())
            pv_by_bidder = pv_payload["pv_by_bidder"]

        bundle_rows = build_bundle_level_rows(benchmark, pv_by_bidder)
        summary_rows = build_summary_rows(
            benchmark,
            pv_by_bidder,
            bundle_rows,
            large_bundle_size_threshold=args.large_bundle_size_threshold,
        )
        all_bundle_rows.extend(bundle_rows)
        all_summary_rows.extend(summary_rows)

    by_size_rows = build_by_bundle_size_rows(all_bundle_rows)
    report = build_report_markdown(benchmarks, all_summary_rows, by_size_rows)

    write_csv(all_bundle_rows, output_dir / "value_calibration_bundle_level.csv")
    write_csv(all_summary_rows, output_dir / "value_calibration_summary.csv")
    write_csv(by_size_rows, output_dir / "value_calibration_by_bundle_size.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "value_calibration_report.md").write_text(report)

    print(f"Wrote diagnostics to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:  # pragma: no cover - argparse enforces valid modes
        raise SystemExit(f"unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
