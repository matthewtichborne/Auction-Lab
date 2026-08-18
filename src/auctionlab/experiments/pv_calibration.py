"""Out-of-domain benchmark and offline fitting for provisional-value calibration.

The provisional-valuation (PV) estimator is the proxy LLM prompt built by
:func:`~auctionlab.llm.prompts.build_provisional_valuation_prompt`. This module
measures that estimator against hidden ground truth in five **non-PC-build**
consumer-bundle domains, and fits a
:class:`~auctionlab.llm.value_calibration.ValueCalibration` to the result.

Keeping the calibration domains disjoint from the PC-build experiment is the
whole point: a scale fitted on the experimental instances would be an in-sample
correction dressed up as a model property. Leave-one-domain-out
cross-validation over the five domains is what licenses (or refuses) the claim
that a fitted calibration generalises.

Three parts live here:

*Benchmark environments* -- deterministic, seeded, not LLM-generated. Each
bidder has a hidden valuation table plus a short qualitative disclosure
rendered by
:func:`~auctionlab.instances.structured.render_brief_qualitative_person_seed`,
which exposes priorities, alternatives, complements, exclusions and exactly one
overall budget, and **no** item-level prices, base values, or synergy bonuses.

*Frozen artefacts* -- one JSON file per (domain, seed) holding hidden truth,
raw provisional predictions, the selected candidate bundles, model provenance,
prompt hashes and call metadata. Preparation makes LLM calls; everything
downstream reads these files and makes none.

*Fitting* -- a deterministic grid search over ``scale``/``size_gamma`` against a
scale-aware objective, with the disclosed-budget cap applied exactly as the
runtime applies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario
from auctionlab.instances.structured import (
    BidderPreferenceProfile,
    ComplementGroup,
    SubstituteGroup,
    _comp_if_available,
    _jitter,
    _sub_if_available,
    generate_full_valuations,
    render_brief_qualitative_person_seed,
)
from auctionlab.llm.value_calibration import (
    CALIBRATION_FAMILIES,
    CalibrationConfigError,
    ValueCalibration,
)


PV_CALIBRATION_BENCHMARK_FORMAT = "auctionlab.pv_calibration_benchmark"
PV_CALIBRATION_BENCHMARK_VERSION = 1

#: The five out-of-domain calibration domains, in canonical order.
BENCHMARK_DOMAINS: tuple[str, ...] = (
    "home_office",
    "travel_package",
    "camera_video_kit",
    "kitchen_appliance_bundle",
    "gaming_peripherals",
)

FITTING_OBJECTIVES: tuple[str, ...] = (
    "budget_normalized_mae",
    "robust_log_error",
)

#: Additive floor used by the log-ratio objective so exact zeros are safe.
LOG_ERROR_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Domain catalogues
# ---------------------------------------------------------------------------
# Plain data: a goods catalogue plus bidder archetypes. Numbers here define the
# *hidden* valuation model only -- they are never rendered into a person or
# proxy prompt. Archetypes deliberately leave some goods out of `base_values`
# so every bidder has genuine exclusions, and each domain carries at least one
# natural substitute pair so the elicited interest map has alternatives to
# find.

DOMAIN_CATALOGS: dict[str, dict[str, Any]] = {
    "home_office": {
        "scenario_description": (
            "A home-office equipment auction. Bidders are furnishing or "
            "upgrading a home workspace and have varying needs for comfort, "
            "video-call quality, and desk ergonomics."
        ),
        "goods": {
            "DESK": "Adjustable-height desk with cable management.",
            "CHAIR_ERGONOMIC": "Ergonomic office chair with full lumbar and arm adjustment.",
            "CHAIR_BASIC": "Basic padded task chair with fixed arms.",
            "MONITOR": "27-inch 1440p monitor with an adjustable stand.",
            "WEBCAM": "1080p webcam with a built-in privacy shutter.",
            "DOCKING_STATION": "USB-C docking station for a single-cable laptop setup.",
        },
        "bidders": [
            {
                "bidder_id": "remote_worker",
                "role": (
                    "Priya works from home full-time and wants a comfortable, "
                    "well-equipped everyday workspace."
                ),
                "budget_range": (400.0, 900.0),
                "base_values": {
                    "DESK": 220.0, "CHAIR_ERGONOMIC": 260.0,
                    "CHAIR_BASIC": 110.0, "MONITOR": 180.0,
                    "DOCKING_STATION": 90.0,
                },
                "substitute_groups": [
                    (frozenset({"CHAIR_ERGONOMIC", "CHAIR_BASIC"}), 0.1,
                     "only one chair fits under the desk", "choose_one"),
                ],
                "complement_groups": [
                    (frozenset({"DESK", "CHAIR_ERGONOMIC", "MONITOR"}), 80.0,
                     "a complete, comfortable everyday workstation"),
                ],
                "core_items": frozenset({"DESK", "CHAIR_ERGONOMIC", "MONITOR"}),
                "secondary_items": frozenset({"DOCKING_STATION"}),
                "low_interest_items": frozenset({"CHAIR_BASIC"}),
            },
            {
                "bidder_id": "video_call_manager",
                "role": (
                    "Jordan runs back-to-back video calls all day and cares "
                    "most about camera quality and how the desk reads on screen."
                ),
                "budget_range": (250.0, 600.0),
                "base_values": {
                    "WEBCAM": 150.0, "MONITOR": 140.0,
                    "CHAIR_ERGONOMIC": 120.0, "CHAIR_BASIC": 55.0,
                    "DOCKING_STATION": 70.0,
                },
                "substitute_groups": [
                    (frozenset({"CHAIR_ERGONOMIC", "CHAIR_BASIC"}), 0.15,
                     "one chair is enough", "choose_one"),
                ],
                "complement_groups": [
                    (frozenset({"WEBCAM", "MONITOR"}), 45.0,
                     "a good camera next to a good screen makes calls work"),
                ],
                "core_items": frozenset({"WEBCAM", "MONITOR"}),
                "secondary_items": frozenset({"CHAIR_ERGONOMIC", "DOCKING_STATION"}),
                "low_interest_items": frozenset({"CHAIR_BASIC"}),
            },
            {
                "bidder_id": "minimalist_freelancer",
                "role": (
                    "Sam freelances from a small apartment and wants the "
                    "fewest items that still make the desk usable."
                ),
                "budget_range": (150.0, 350.0),
                "base_values": {
                    "DESK": 130.0, "CHAIR_BASIC": 95.0,
                    "CHAIR_ERGONOMIC": 130.0, "MONITOR": 90.0,
                },
                "substitute_groups": [
                    (frozenset({"CHAIR_ERGONOMIC", "CHAIR_BASIC"}), 0.05,
                     "there is only room for one chair", "choose_one"),
                ],
                "saturation_start": 3,
                "saturation_penalty": 12.0,
                "core_items": frozenset({"DESK", "CHAIR_ERGONOMIC"}),
                "secondary_items": frozenset({"MONITOR"}),
                "low_interest_items": frozenset({"CHAIR_BASIC"}),
            },
            {
                "bidder_id": "office_reseller",
                "role": (
                    "Morgan refurbishes and resells office furniture and can "
                    "move more than one of most things."
                ),
                "budget_range": (500.0, 1000.0),
                "base_values": {
                    "CHAIR_ERGONOMIC": 240.0, "CHAIR_BASIC": 130.0,
                    "DESK": 200.0, "MONITOR": 130.0,
                    "DOCKING_STATION": 70.0, "WEBCAM": 45.0,
                },
                "substitute_groups": [
                    (frozenset({"CHAIR_ERGONOMIC", "CHAIR_BASIC"}), 0.8,
                     "both chairs can be resold separately", "can_use_multiple"),
                ],
                "core_items": frozenset({"CHAIR_ERGONOMIC", "DESK"}),
                "secondary_items": frozenset({"CHAIR_BASIC", "MONITOR"}),
                "low_interest_items": frozenset({"DOCKING_STATION", "WEBCAM"}),
            },
        ],
    },
    "travel_package": {
        "scenario_description": (
            "A bundled travel-package auction. Bidders are booking a trip "
            "and have different priorities around comfort, flexibility, "
            "and cost."
        ),
        "goods": {
            "FLIGHT_DIRECT": "Round-trip direct flight in economy.",
            "FLIGHT_CONNECTING": "Cheaper round-trip flight with one connection.",
            "HOTEL": "Four-night hotel stay in the destination city.",
            "CAR_RENTAL": "Compact car rental for the duration of the trip.",
            "TRAVEL_INSURANCE": "Trip cancellation and medical travel insurance.",
            "GUIDED_TOUR": "A half-day guided city tour with a local guide.",
        },
        "bidders": [
            {
                "bidder_id": "family_vacationer",
                "role": (
                    "The Alvarez family is planning a relaxed one-week "
                    "vacation and values convenience over cost."
                ),
                "budget_range": (1200.0, 2200.0),
                "base_values": {
                    "FLIGHT_DIRECT": 700.0, "FLIGHT_CONNECTING": 480.0,
                    "HOTEL": 700.0, "CAR_RENTAL": 220.0,
                    "TRAVEL_INSURANCE": 90.0, "GUIDED_TOUR": 140.0,
                },
                "substitute_groups": [
                    (frozenset({"FLIGHT_DIRECT", "FLIGHT_CONNECTING"}), 0.05,
                     "the family flies out once", "choose_one"),
                ],
                "complement_groups": [
                    (frozenset({"FLIGHT_DIRECT", "HOTEL", "CAR_RENTAL"}), 130.0,
                     "a fully-booked trip removes all logistics stress"),
                ],
                "core_items": frozenset({"FLIGHT_DIRECT", "HOTEL", "CAR_RENTAL"}),
                "secondary_items": frozenset({"TRAVEL_INSURANCE", "GUIDED_TOUR"}),
                "low_interest_items": frozenset({"FLIGHT_CONNECTING"}),
            },
            {
                "bidder_id": "business_traveler",
                "role": (
                    "Dana is on a tight one-night business trip and cares "
                    "most about arriving on time and without hassle."
                ),
                "budget_range": (600.0, 1400.0),
                "base_values": {
                    "FLIGHT_DIRECT": 560.0, "FLIGHT_CONNECTING": 200.0,
                    "HOTEL": 300.0, "CAR_RENTAL": 150.0,
                },
                "substitute_groups": [
                    (frozenset({"FLIGHT_DIRECT", "FLIGHT_CONNECTING"}), 0.05,
                     "only one outbound itinerary is usable", "choose_one"),
                ],
                "complement_groups": [
                    (frozenset({"FLIGHT_DIRECT", "HOTEL"}), 70.0,
                     "flight and hotel booked together make the trip work at all"),
                ],
                "core_items": frozenset({"FLIGHT_DIRECT", "HOTEL"}),
                "secondary_items": frozenset({"CAR_RENTAL"}),
                "low_interest_items": frozenset({"FLIGHT_CONNECTING"}),
            },
            {
                "bidder_id": "budget_backpacker",
                "role": (
                    "Riley is backpacking on a strict budget and only wants "
                    "the essentials, skipping anything that feels optional."
                ),
                "budget_range": (300.0, 700.0),
                "base_values": {
                    "FLIGHT_CONNECTING": 330.0, "FLIGHT_DIRECT": 260.0,
                    "HOTEL": 170.0, "GUIDED_TOUR": 70.0,
                    "TRAVEL_INSURANCE": 45.0,
                },
                "substitute_groups": [
                    (frozenset({"FLIGHT_DIRECT", "FLIGHT_CONNECTING"}), 0.05,
                     "one flight, whichever is cheaper", "choose_one"),
                ],
                "saturation_start": 3,
                "saturation_penalty": 25.0,
                "core_items": frozenset({"FLIGHT_CONNECTING", "HOTEL"}),
                "secondary_items": frozenset({"GUIDED_TOUR", "TRAVEL_INSURANCE"}),
                "low_interest_items": frozenset({"FLIGHT_DIRECT"}),
            },
            {
                "bidder_id": "luxury_honeymooner",
                "role": (
                    "Chris and Avery are planning a honeymoon and want "
                    "every part of the trip to feel effortless."
                ),
                "budget_range": (2500.0, 4500.0),
                "base_values": {
                    "FLIGHT_DIRECT": 950.0, "FLIGHT_CONNECTING": 350.0,
                    "HOTEL": 1400.0, "CAR_RENTAL": 300.0,
                    "GUIDED_TOUR": 350.0, "TRAVEL_INSURANCE": 150.0,
                },
                "substitute_groups": [
                    (frozenset({"FLIGHT_DIRECT", "FLIGHT_CONNECTING"}), 0.05,
                     "one itinerary each way", "choose_one"),
                ],
                "complement_groups": [
                    (frozenset({"HOTEL", "GUIDED_TOUR"}), 160.0,
                     "a premium stay paired with a curated tour feels cohesive"),
                ],
                "core_items": frozenset({"HOTEL", "FLIGHT_DIRECT", "GUIDED_TOUR"}),
                "secondary_items": frozenset({"CAR_RENTAL", "TRAVEL_INSURANCE"}),
                "low_interest_items": frozenset({"FLIGHT_CONNECTING"}),
            },
        ],
    },
    "camera_video_kit": {
        "scenario_description": (
            "A camera and video production equipment auction. Bidders are "
            "assembling a kit for different kinds of shooting work."
        ),
        "goods": {
            "CAMERA_BODY": "Mirrorless camera body with in-body stabilization.",
            "LENS_WIDE": "16-35mm wide-angle zoom lens.",
            "LENS_TELEPHOTO": "70-200mm telephoto zoom lens.",
            "TRIPOD": "Carbon-fiber tripod with a fluid video head.",
            "LIGHTING_KIT": "Two-point LED lighting kit with softboxes.",
            "MEMORY_CARDS": "Set of three high-speed memory cards.",
        },
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
                    "MEMORY_CARDS": 150.0,
                },
                "substitute_groups": [
                    (frozenset({"LENS_TELEPHOTO", "LENS_WIDE"}), 0.4,
                     "the wide lens is a distant second choice in the field",
                     "can_use_multiple"),
                ],
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "LENS_TELEPHOTO"}), 220.0,
                     "the body and telephoto lens together are the core wildlife setup"),
                ],
                "core_items": frozenset({"CAMERA_BODY", "LENS_TELEPHOTO"}),
                "secondary_items": frozenset({"TRIPOD", "MEMORY_CARDS"}),
                "low_interest_items": frozenset({"LENS_WIDE"}),
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
                "substitute_groups": [
                    (frozenset({"LENS_WIDE", "LENS_TELEPHOTO"}), 0.7,
                     "both lenses get swapped in over the course of a wedding",
                     "can_use_multiple"),
                ],
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "TRIPOD", "LIGHTING_KIT"}), 260.0,
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
                    "LENS_TELEPHOTO": 130.0, "TRIPOD": 120.0,
                    "LIGHTING_KIT": 150.0,
                },
                "substitute_groups": [
                    (frozenset({"LENS_WIDE", "LENS_TELEPHOTO"}), 0.1,
                     "only one lens is affordable this year", "choose_one"),
                ],
                "saturation_start": 3,
                "saturation_penalty": 30.0,
                "core_items": frozenset({"CAMERA_BODY", "LENS_WIDE"}),
                "secondary_items": frozenset({"LIGHTING_KIT", "TRIPOD"}),
                "low_interest_items": frozenset({"LENS_TELEPHOTO"}),
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
                    "TRIPOD": 200.0, "MEMORY_CARDS": 90.0,
                },
                "complement_groups": [
                    (frozenset({"CAMERA_BODY", "LIGHTING_KIT"}), 200.0,
                     "controlled studio lighting paired with the body defines the look"),
                ],
                "core_items": frozenset({"LIGHTING_KIT", "CAMERA_BODY"}),
                "secondary_items": frozenset({"TRIPOD"}),
                "low_interest_items": frozenset({"MEMORY_CARDS"}),
            },
        ],
    },
    "kitchen_appliance_bundle": {
        "scenario_description": (
            "A kitchen appliance bundle auction. Bidders are outfitting a "
            "kitchen for different styles of home cooking."
        ),
        "goods": {
            "STAND_MIXER": "Stand mixer with dough hook and whisk attachments.",
            "BLENDER": "High-power countertop blender.",
            "AIR_FRYER": "Six-quart digital air fryer.",
            "COFFEE_MACHINE": "Espresso machine with a built-in grinder.",
            "SOUS_VIDE": "Immersion sous vide precision cooker.",
            "FOOD_PROCESSOR": "Multi-blade food processor with slicing discs.",
        },
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
                    "BLENDER": 90.0,
                },
                "complement_groups": [
                    (frozenset({"STAND_MIXER", "FOOD_PROCESSOR"}), 60.0,
                     "mixing and prep together cover nearly all baking tasks"),
                ],
                "core_items": frozenset({"STAND_MIXER", "FOOD_PROCESSOR"}),
                "secondary_items": frozenset({"BLENDER"}),
                "low_interest_items": frozenset(),
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
                },
                "substitute_groups": [
                    (frozenset({"BLENDER", "FOOD_PROCESSOR"}), 0.35,
                     "either one handles the prep, with some overlap",
                     "can_use_multiple"),
                ],
                "complement_groups": [
                    (frozenset({"SOUS_VIDE", "AIR_FRYER"}), 50.0,
                     "sous-vide followed by an air-fryer sear covers most weeknight meals"),
                ],
                "core_items": frozenset({"SOUS_VIDE", "AIR_FRYER"}),
                "secondary_items": frozenset({"BLENDER", "FOOD_PROCESSOR"}),
                "low_interest_items": frozenset(),
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
                    "FOOD_PROCESSOR": 110.0, "COFFEE_MACHINE": 90.0,
                },
                "substitute_groups": [
                    (frozenset({"BLENDER", "FOOD_PROCESSOR"}), 0.15,
                     "only one more appliance fits on the counter", "choose_one"),
                ],
                "saturation_start": 3,
                "saturation_penalty": 18.0,
                "core_items": frozenset({"AIR_FRYER", "BLENDER"}),
                "secondary_items": frozenset({"COFFEE_MACHINE"}),
                "low_interest_items": frozenset({"FOOD_PROCESSOR"}),
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
                },
                "core_items": frozenset({"COFFEE_MACHINE"}),
                "secondary_items": frozenset({"BLENDER"}),
                "low_interest_items": frozenset(),
            },
        ],
    },
    "gaming_peripherals": {
        "scenario_description": (
            "A gaming peripherals auction. Bidders are equipping different "
            "gaming setups, from competitive PC play to relaxed console play."
        ),
        "goods": {
            "MECHANICAL_KEYBOARD": "Hot-swappable mechanical keyboard.",
            "GAMING_MOUSE": "Lightweight wireless gaming mouse.",
            "HEADSET": "Wireless gaming headset with a detachable microphone.",
            "MONITOR_144HZ": "27-inch 144Hz gaming monitor.",
            "CONTROLLER": "Wireless controller with remappable back buttons.",
            "MOUSE_PAD": "Extended desk-size mouse pad.",
        },
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
                    "HEADSET": 120.0,
                },
                "complement_groups": [
                    (frozenset({"MONITOR_144HZ", "GAMING_MOUSE", "MECHANICAL_KEYBOARD"}), 80.0,
                     "a matched high-refresh monitor and low-latency mouse and "
                     "keyboard is the whole point of a competitive setup"),
                ],
                "core_items": frozenset(
                    {"MONITOR_144HZ", "GAMING_MOUSE", "MECHANICAL_KEYBOARD"}
                ),
                "secondary_items": frozenset({"HEADSET", "MOUSE_PAD"}),
                "low_interest_items": frozenset(),
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
                    "MONITOR_144HZ": 90.0,
                },
                "complement_groups": [
                    (frozenset({"CONTROLLER", "HEADSET"}), 35.0,
                     "a comfortable controller and a good headset are used together "
                     "every session"),
                ],
                "core_items": frozenset({"CONTROLLER", "HEADSET"}),
                "secondary_items": frozenset({"MONITOR_144HZ"}),
                "low_interest_items": frozenset(),
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
                "substitute_groups": [
                    (frozenset({"GAMING_MOUSE", "CONTROLLER"}), 0.6,
                     "different games on stream call for different input devices",
                     "can_use_multiple"),
                ],
                "complement_groups": [
                    (frozenset({"MECHANICAL_KEYBOARD", "MOUSE_PAD"}), 30.0,
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
                    "CONTROLLER": 55.0,
                },
                "substitute_groups": [
                    (frozenset({"GAMING_MOUSE", "CONTROLLER"}), 0.1,
                     "one input device is plenty", "choose_one"),
                ],
                "saturation_start": 3,
                "saturation_penalty": 10.0,
                "core_items": frozenset({"HEADSET", "MONITOR_144HZ"}),
                "secondary_items": frozenset({"GAMING_MOUSE", "MECHANICAL_KEYBOARD"}),
                "low_interest_items": frozenset({"CONTROLLER"}),
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Deterministic benchmark environments
# ---------------------------------------------------------------------------

def domain_goods(domain: str) -> list[str]:
    return list(_catalog(domain)["goods"])


def domain_bidder_ids(domain: str) -> list[str]:
    return [archetype["bidder_id"] for archetype in _catalog(domain)["bidders"]]


def _catalog(domain: str) -> dict[str, Any]:
    try:
        return DOMAIN_CATALOGS[domain]
    except KeyError:
        raise ValueError(
            f"unknown benchmark domain {domain!r}; available: "
            f"{list(BENCHMARK_DOMAINS)}"
        ) from None


def _build_profile(
    archetype: Mapping[str, Any],
    items: set[str],
    rng: random.Random,
) -> BidderPreferenceProfile:
    base_values = {
        item: _jitter(value, rng)
        for item, value in archetype["base_values"].items()
        if item in items
    }
    substitute_groups: list[SubstituteGroup] = []
    for group_items, backup_factor, description, mode in archetype.get(
        "substitute_groups", []
    ):
        substitute_groups += _sub_if_available(
            items,
            group_items,
            backup_factor,
            description,
            acquisition_mode=mode,
        )
    complement_groups: list[ComplementGroup] = []
    for group_items, bonus, description in archetype.get("complement_groups", []):
        complement_groups += _comp_if_available(
            items, group_items, _jitter(bonus, rng), description
        )
    return BidderPreferenceProfile(
        bidder_id=archetype["bidder_id"],
        role=archetype["role"],
        budget_range=tuple(archetype["budget_range"]),
        base_values=base_values,
        substitute_groups=substitute_groups,
        complement_groups=complement_groups,
        saturation_start=archetype.get("saturation_start"),
        saturation_penalty=archetype.get("saturation_penalty", 0.0),
        core_items=frozenset(archetype.get("core_items", frozenset())) & items,
        secondary_items=(
            frozenset(archetype.get("secondary_items", frozenset())) & items
        ),
        low_interest_items=(
            frozenset(archetype.get("low_interest_items", frozenset())) & items
        ),
    )


def build_benchmark_scenario(
    domain: str,
    *,
    seed: int = 0,
    num_goods: int | None = None,
    num_bidders: int | None = None,
) -> NaturalLanguageAuctionScenario:
    """Build one deterministic, fully-specified benchmark environment.

    Identical inputs always produce an identical scenario: the only source of
    randomness is a ``random.Random(seed)`` consumed in a fixed order by the
    ±5% jitter applied to hidden base values and complement bonuses.

    The environment is hand-authored rather than LLM-generated on purpose --
    the benchmark exists to calibrate the value estimator, and an LLM-generated
    environment would put a second, uncontrolled model between the estimator
    and the ground truth it is being scored against.
    """
    catalog = _catalog(domain)
    all_goods = list(catalog["goods"])
    all_archetypes = list(catalog["bidders"])
    num_goods = len(all_goods) if num_goods is None else num_goods
    num_bidders = len(all_archetypes) if num_bidders is None else num_bidders

    if not 1 <= num_goods <= len(all_goods):
        raise ValueError(
            f"num_goods must be between 1 and {len(all_goods)} for domain "
            f"{domain!r}, got {num_goods}"
        )
    if not 1 <= num_bidders <= len(all_archetypes):
        raise ValueError(
            f"num_bidders must be between 1 and {len(all_archetypes)} for "
            f"domain {domain!r}, got {num_bidders}"
        )

    items = all_goods[:num_goods]
    items_set = set(items)
    archetypes = all_archetypes[:num_bidders]

    rng = random.Random(seed)
    profiles = [_build_profile(a, items_set, rng) for a in archetypes]
    valuations = generate_full_valuations(items, profiles)

    person_seeds = {
        profile.bidder_id: render_brief_qualitative_person_seed(
            profile,
            identity_text=profile.role,
            available_goods=items,
        )
        for profile in profiles
    }
    profile_metadata = {
        profile.bidder_id: {
            "role": profile.role,
            "budget_range": list(profile.budget_range),
            "disclosed_budget_hint": max(
                valuations[profile.bidder_id].values(), default=0.0
            ),
            "disclosed_positive_items": sorted(
                item for item in items if profile.base_values.get(item, 0.0) > 0
            ),
            "core_items": sorted(profile.core_items),
            "secondary_items": sorted(profile.secondary_items),
            "low_interest_items": sorted(profile.low_interest_items),
            "substitute_groups": [
                {
                    "items": sorted(sg.items),
                    "backup_factor": sg.backup_factor,
                    "acquisition_mode": sg.acquisition_mode,
                }
                for sg in profile.substitute_groups
            ],
            "complement_groups": [
                {"items": sorted(cg.items), "bonus": cg.bonus}
                for cg in profile.complement_groups
            ],
            "person_seed_source": "brief_qualitative_disclosure",
            "person_seed_identity_source": "role",
        }
        for profile in profiles
    }

    return NaturalLanguageAuctionScenario(
        name=f"pv_calib_{domain}_{num_goods}x{num_bidders}_seed{seed}",
        seed_type="structured",
        instance=AuctionInstance(
            items=items,
            bidder_ids=[p.bidder_id for p in profiles],
            valuations=valuations,
        ),
        scenario_description=catalog["scenario_description"],
        item_descriptions={item: catalog["goods"][item] for item in items},
        person_seeds=person_seeds,
        candidate_bundles_by_bidder=None,
        metadata={
            "num_goods": num_goods,
            "num_bidders": num_bidders,
            "scenario_seed": seed,
            "domain": domain,
            "benchmark": "pv_calibration",
            "valuation_model": "structured_substitutes_complements",
            "seed_style": "brief_qualitative",
            "profiles": profile_metadata,
        },
    )


# ---------------------------------------------------------------------------
# Frozen benchmark artefacts
# ---------------------------------------------------------------------------

def _bundle_list(bundle: frozenset[str]) -> list[str]:
    return sorted(bundle)


def bundle_key(bundle: Iterable[str]) -> str:
    """Stable bundle serialisation: ``+``-joined sorted item ids."""
    return "+".join(sorted(bundle))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PvObservation:
    """One (bidder, bundle) prediction paired with its hidden true value."""

    domain: str
    seed: int
    bidder_id: str
    bundle: frozenset[str]
    raw_value: float
    true_value: float
    disclosed_budget: float | None

    @property
    def bundle_size(self) -> int:
        return len(self.bundle)


def write_benchmark_artefact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_benchmark_artefact(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != PV_CALIBRATION_BENCHMARK_FORMAT:
        raise ValueError(
            f"{path}: not a pv-calibration benchmark artefact "
            f"(format={data.get('format')!r})"
        )
    if int(data.get("version", 0)) != PV_CALIBRATION_BENCHMARK_VERSION:
        raise ValueError(
            f"{path}: unsupported artefact version {data.get('version')!r}; "
            f"expected {PV_CALIBRATION_BENCHMARK_VERSION}"
        )
    bidders = data.get("bidders")
    if not isinstance(bidders, dict) or not bidders:
        raise ValueError(
            f"{path}: benchmark artefact has no non-empty bidders mapping"
        )
    return data


def artefact_file_name(domain: str, seed: int) -> str:
    return f"pv_calibration_{domain}_seed{seed}.json"


def observations_from_artefact(
    artefact: Mapping[str, Any],
) -> list[PvObservation]:
    """Flatten one frozen artefact into per-bundle observations.

    Bundles the proxy failed to price are dropped rather than defaulted to
    zero: a missing prediction is missing data, and scoring it as a zero
    prediction would bias the fitted scale upwards.
    """
    domain = str(artefact["domain"])
    seed = int(artefact["seed"])
    observations: list[PvObservation] = []
    for bidder_id, entry in sorted(artefact["bidders"].items()):
        truth = {
            bundle_key(row["bundle"]): float(row["value"])
            for row in entry["hidden_true_values"]
        }
        budget = entry.get("disclosed_budget")
        for row in entry.get("raw_provisional_values") or []:
            key = bundle_key(row["bundle"])
            if key not in truth:
                continue
            observations.append(
                PvObservation(
                    domain=domain,
                    seed=seed,
                    bidder_id=str(bidder_id),
                    bundle=frozenset(row["bundle"]),
                    raw_value=float(row["value"]),
                    true_value=truth[key],
                    disclosed_budget=(
                        None if budget is None else float(budget)
                    ),
                )
            )
    return observations


def load_observations(
    paths: Sequence[str | Path],
) -> tuple[list[PvObservation], dict[str, str]]:
    """Load every artefact and return its observations plus content hashes."""
    observations: list[PvObservation] = []
    hashes: dict[str, str] = {}
    for path in paths:
        artefact = load_benchmark_artefact(path)
        observations.extend(observations_from_artefact(artefact))
        hashes[str(path)] = file_sha256(path)
    return observations, hashes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _budget_norm(observation: PvObservation) -> float:
    """Denominator for budget-normalised error, never zero.

    Normalising by the disclosed budget rather than by the true value keeps
    the objective on a scale the calibration is entitled to know: the budget
    is disclosed, whereas the true value is exactly what is being estimated.
    It also stops cheap bundles dominating the average, since a small
    absolute error on a low-value bundle would otherwise look enormous in
    relative terms. Falling back to the true value only arises when no
    budget was disclosed.
    """
    if observation.disclosed_budget and observation.disclosed_budget > 0:
        return float(observation.disclosed_budget)
    return max(observation.true_value, LOG_ERROR_FLOOR)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman_rank_correlation(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def top_k_recall(
    predicted: Mapping[str, float],
    truth: Mapping[str, float],
    k: int,
) -> float | None:
    keys = [key for key in truth if key in predicted]
    if k <= 0 or len(keys) < k:
        return None
    true_top = set(sorted(keys, key=lambda b: (-truth[b], b))[:k])
    pred_top = set(sorted(keys, key=lambda b: (-predicted[b], b))[:k])
    return len(true_top & pred_top) / k


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def evaluate_predictions(
    observations: Sequence[PvObservation],
    calibration: ValueCalibration,
    *,
    top_k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Score one calibration over a set of observations.

    Reports absolute, budget-normalised and log-scale error side by side
    because they disagree in informative ways: MAE is dominated by the
    highest-value bidders, budget-normalised MAE puts bidders on a common
    footing, and the log error is the only one that treats a 2x
    under-estimate and a 2x over-estimate symmetrically.
    """
    if not observations:
        return {"n": 0}

    predicted = [predict(observation, calibration) for observation in observations]
    truths = [observation.true_value for observation in observations]
    signed = [p - t for p, t in zip(predicted, truths)]
    absolute = [abs(value) for value in signed]

    metrics: dict[str, Any] = {
        "n": len(observations),
        "mae": _mean(absolute),
        "rmse": math.sqrt(_mean([value * value for value in signed])),
        "budget_normalized_mae": _mean(
            [
                abs(p - t) / _budget_norm(observation)
                for p, t, observation in zip(predicted, truths, observations)
            ]
        ),
        "robust_log_error": _mean(
            [
                abs(
                    math.log((p + LOG_ERROR_FLOOR) / (t + LOG_ERROR_FLOOR))
                )
                for p, t in zip(predicted, truths)
            ]
        ),
        "signed_bias": _mean(signed),
        "budget_normalized_signed_bias": _mean(
            [
                (p - t) / _budget_norm(observation)
                for p, t, observation in zip(predicted, truths, observations)
            ]
        ),
        "median_true_over_predicted": statistics.median(
            [
                (t + LOG_ERROR_FLOOR) / (p + LOG_ERROR_FLOOR)
                for p, t in zip(predicted, truths)
            ]
        ),
    }

    metrics["spearman"] = spearman_rank_correlation(predicted, truths)

    # Rank quality is a per-bidder property: pooling bundles across bidders
    # with different value scales would measure scale agreement, not ranking.
    by_bidder: dict[tuple[str, int, str], list[tuple[str, float, float]]] = {}
    for observation, prediction in zip(observations, predicted):
        key = (observation.domain, observation.seed, observation.bidder_id)
        by_bidder.setdefault(key, []).append(
            (bundle_key(observation.bundle), prediction, observation.true_value)
        )
    per_bidder_spearman: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in top_k_values}
    for rows in by_bidder.values():
        correlation = spearman_rank_correlation(
            [row[1] for row in rows], [row[2] for row in rows]
        )
        if correlation is not None:
            per_bidder_spearman.append(correlation)
        predicted_map = {row[0]: row[1] for row in rows}
        truth_map = {row[0]: row[2] for row in rows}
        for k in top_k_values:
            recall = top_k_recall(predicted_map, truth_map, k)
            if recall is not None:
                recalls[k].append(recall)
    metrics["mean_bidder_spearman"] = (
        _mean(per_bidder_spearman) if per_bidder_spearman else None
    )
    for k in top_k_values:
        metrics[f"top_{k}_recall"] = (
            _mean(recalls[k]) if recalls[k] else None
        )
    return metrics


def errors_by_bundle_size(
    observations: Sequence[PvObservation],
    calibration: ValueCalibration,
) -> list[dict[str, Any]]:
    """Per-bundle-size error breakdown; the size-gamma diagnostic."""
    grouped: dict[int, list[PvObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.bundle_size, []).append(observation)
    rows: list[dict[str, Any]] = []
    for size in sorted(grouped):
        group = grouped[size]
        predicted = [predict(observation, calibration) for observation in group]
        rows.append(
            {
                "bundle_size": size,
                "n": len(group),
                "mae": _mean([abs(p - o.true_value) for p, o in zip(predicted, group)]),
                "signed_bias": _mean(
                    [p - o.true_value for p, o in zip(predicted, group)]
                ),
                "budget_normalized_mae": _mean(
                    [
                        abs(p - o.true_value) / _budget_norm(o)
                        for p, o in zip(predicted, group)
                    ]
                ),
                "median_true_over_predicted": statistics.median(
                    [
                        (o.true_value + LOG_ERROR_FLOOR) / (p + LOG_ERROR_FLOOR)
                        for p, o in zip(predicted, group)
                    ]
                ),
            }
        )
    return rows


def predict(
    observation: PvObservation,
    calibration: ValueCalibration,
) -> float:
    """Apply a calibration to one observation exactly as the runtime does.

    Routed through ``ValueCalibration.apply`` rather than reimplementing the
    formula, so that a scale selected here cannot be fitted under slightly
    different arithmetic from the one the auction will use. In particular the
    disclosed-budget cap is applied at fitting time too: a scale chosen
    without it would be tuned against values the mechanism never sees.
    """
    return calibration.apply(
        observation.raw_value,
        observation.bundle_size,
        disclosed_budget=observation.disclosed_budget,
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def objective_value(
    observations: Sequence[PvObservation],
    calibration: ValueCalibration,
    objective: str,
) -> float:
    if objective not in FITTING_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {list(FITTING_OBJECTIVES)}, "
            f"got {objective!r}"
        )
    if not observations:
        return float("inf")
    total = 0.0
    for observation in observations:
        prediction = predict(observation, calibration)
        if objective == "budget_normalized_mae":
            total += abs(prediction - observation.true_value) / _budget_norm(
                observation
            )
        else:
            total += abs(
                math.log(
                    (prediction + LOG_ERROR_FLOOR)
                    / (observation.true_value + LOG_ERROR_FLOOR)
                )
            )
    return total / len(observations)


def size_gamma_is_identifiable(
    observations: Sequence[PvObservation],
    size_threshold: int,
) -> bool:
    """True when at least one observed bundle exceeds ``size_threshold``.

    Below the threshold the size factor is exactly ``gamma ** 0 == 1`` for
    every observation, so the objective is flat in ``gamma`` and any value
    fits equally well. Reporting the grid's arbitrary pick as a fitted size
    effect would be a fabricated finding, so the fitter pins gamma to 1
    instead.
    """
    return any(
        observation.bundle_size > size_threshold for observation in observations
    )


@dataclass(frozen=True)
class FitResult:
    """One fitted calibration plus the objective it achieved."""

    calibration: ValueCalibration
    objective: str
    objective_value: float
    n_observations: int
    size_threshold_grid: tuple[int, ...] = ()
    threshold_scores: tuple[tuple[int, float], ...] = ()
    size_gamma_identifiable: bool = True
    n_above_threshold: int = 0


def _grid_search(
    observations: Sequence[PvObservation],
    *,
    objective: str,
    family: str,
    size_threshold: int,
    budget_cap: bool,
    scale_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    steps: int = 25,
    passes: int = 5,
) -> tuple[float, float, float]:
    """Deterministic coarse-to-fine search over (scale, size_gamma).

    A grid rather than a gradient method because the disclosed-budget cap
    makes the objective piecewise-linear with kinks, and because a fixed grid
    is reproducible to the last digit across machines with no extra
    dependency. Each pass shrinks the window around the incumbent optimum.
    """
    scale_lo, scale_hi = scale_bounds
    gamma_lo, gamma_hi = gamma_bounds
    fit_gamma = family == "exponential" and size_gamma_is_identifiable(
        observations, size_threshold
    )
    best = (1.0, 1.0, float("inf"))

    for _ in range(passes):
        scale_grid = _linspace(scale_lo, scale_hi, steps)
        gamma_grid = _linspace(gamma_lo, gamma_hi, steps) if fit_gamma else [1.0]
        for scale in scale_grid:
            for gamma in gamma_grid:
                candidate = ValueCalibration(
                    family=family,
                    scale=scale,
                    size_gamma=gamma,
                    size_threshold=size_threshold,
                    budget_cap=budget_cap,
                )
                score = objective_value(observations, candidate, objective)
                # Ties are broken towards gamma = 1 (no size effect), so a
                # flat objective reports "no evidence" instead of an
                # arbitrary grid point that looks like a finding.
                improved = score < best[2] - 1e-12
                tied_and_tamer = (
                    abs(score - best[2]) <= 1e-12
                    and abs(math.log(gamma)) < abs(math.log(best[1]))
                )
                if improved or tied_and_tamer:
                    best = (scale, gamma, score)
        scale_step = (scale_hi - scale_lo) / (steps - 1)
        gamma_step = (gamma_hi - gamma_lo) / (steps - 1) if fit_gamma else 0.0
        scale_lo = max(1e-6, best[0] - scale_step)
        scale_hi = best[0] + scale_step
        if fit_gamma:
            gamma_lo = max(1e-6, best[1] - gamma_step)
            gamma_hi = best[1] + gamma_step
    return best


def _linspace(low: float, high: float, steps: int) -> list[float]:
    if steps < 2 or high <= low:
        return [low]
    span = (high - low) / (steps - 1)
    return [low + span * index for index in range(steps)]


def fit_calibration(
    observations: Sequence[PvObservation],
    *,
    objective: str = "budget_normalized_mae",
    family: str = "exponential",
    size_threshold: int = 3,
    size_threshold_grid: Sequence[int] | None = None,
    budget_cap: bool = True,
    scale_bounds: tuple[float, float] = (0.2, 5.0),
    gamma_bounds: tuple[float, float] = (0.6, 1.4),
    steps: int = 25,
    passes: int = 5,
) -> FitResult:
    """Fit ``scale`` (and ``size_gamma`` for the exponential family).

    ``size_threshold`` is held fixed at 3 by default -- the smallest
    assumption that still lets a size effect exist, and the value the legacy
    interface used. Passing ``size_threshold_grid`` selects the threshold by
    the same objective; because that is one more parameter chosen on the
    benchmark, callers should cross-validate it rather than read the in-sample
    winner (``scripts/fit_pv_calibration.py`` does).
    """
    if family not in CALIBRATION_FAMILIES:
        raise CalibrationConfigError(
            f"family must be one of {list(CALIBRATION_FAMILIES)}, got {family!r}"
        )
    if not observations:
        raise ValueError("cannot fit a calibration with no observations")

    if family == "none":
        calibration = ValueCalibration(family="none")
        return FitResult(
            calibration=calibration,
            objective=objective,
            objective_value=objective_value(observations, calibration, objective),
            n_observations=len(observations),
        )

    thresholds = (
        [size_threshold]
        if size_threshold_grid is None
        else sorted({int(value) for value in size_threshold_grid})
    )
    scored: list[tuple[int, float, float, float]] = []
    for threshold in thresholds:
        scale, gamma, score = _grid_search(
            observations,
            objective=objective,
            family=family,
            size_threshold=threshold,
            budget_cap=budget_cap,
            scale_bounds=scale_bounds,
            gamma_bounds=gamma_bounds,
            steps=steps,
            passes=passes,
        )
        scored.append((threshold, scale, gamma, score))

    threshold, scale, gamma, score = min(scored, key=lambda row: (row[3], row[0]))
    above = sum(
        1 for observation in observations if observation.bundle_size > threshold
    )
    return FitResult(
        calibration=ValueCalibration(
            family=family,
            scale=scale,
            size_gamma=gamma if family == "exponential" else 1.0,
            size_threshold=threshold,
            budget_cap=budget_cap,
        ),
        objective=objective,
        objective_value=score,
        n_observations=len(observations),
        size_threshold_grid=tuple(thresholds),
        threshold_scores=tuple((row[0], row[3]) for row in scored),
        size_gamma_identifiable=size_gamma_is_identifiable(observations, threshold),
        n_above_threshold=above,
    )


@dataclass(frozen=True)
class FoldResult:
    """One leave-one-domain-out fold."""

    held_out_domain: str
    fit: FitResult
    train_metrics: dict[str, Any]
    test_metrics_raw: dict[str, Any]
    test_metrics_calibrated: dict[str, Any]


def leave_one_domain_out(
    observations: Sequence[PvObservation],
    *,
    objective: str = "budget_normalized_mae",
    family: str = "exponential",
    size_threshold: int = 3,
    size_threshold_grid: Sequence[int] | None = None,
    budget_cap: bool = True,
    **fit_kwargs: Any,
) -> list[FoldResult]:
    """Fit on four domains, score on the fifth, for each domain in turn.

    This is the only evidence that speaks to whether a fitted calibration
    transfers to a domain it never saw -- which is exactly the question posed
    by applying it to the PC-build experiment.
    """
    domains = sorted({observation.domain for observation in observations})
    if len(domains) < 2:
        raise ValueError(
            "leave-one-domain-out requires at least two domains, got "
            f"{domains}"
        )
    raw = ValueCalibration(family="none")
    folds: list[FoldResult] = []
    for held_out in domains:
        train = [o for o in observations if o.domain != held_out]
        test = [o for o in observations if o.domain == held_out]
        fit = fit_calibration(
            train,
            objective=objective,
            family=family,
            size_threshold=size_threshold,
            size_threshold_grid=size_threshold_grid,
            budget_cap=budget_cap,
            **fit_kwargs,
        )
        folds.append(
            FoldResult(
                held_out_domain=held_out,
                fit=fit,
                train_metrics=evaluate_predictions(train, fit.calibration),
                test_metrics_raw=evaluate_predictions(test, raw),
                test_metrics_calibrated=evaluate_predictions(
                    test, fit.calibration
                ),
            )
        )
    return folds


# ---------------------------------------------------------------------------
# Synthetic observations (testing and pipeline rehearsal; never a live call)
# ---------------------------------------------------------------------------

def synthesize_observations(
    scenario: NaturalLanguageAuctionScenario,
    *,
    true_calibration: ValueCalibration,
    candidate_bundles_by_bidder: Mapping[str, Sequence[frozenset[str]]] | None = None,
    noise_scale: float = 0.0,
    seed: int = 0,
    disclosed_budgets: Mapping[str, float] | None = None,
) -> list[PvObservation]:
    """Invert a known calibration to fabricate raw predictions.

    Given ``true_calibration`` the raw value is set so that applying that
    calibration recovers the hidden truth, optionally perturbed by symmetric
    multiplicative noise. A fitter run on the result should recover
    ``true_calibration``'s parameters -- which is how the fitter itself is
    tested, with no model and no network involved.
    """
    rng = random.Random(seed)
    domain = str(scenario.metadata.get("domain", scenario.name))
    scenario_seed = int(scenario.metadata.get("scenario_seed", 0))
    observations: list[PvObservation] = []
    for bidder_id in scenario.instance.bidder_ids:
        truth = scenario.instance.valuations[bidder_id]
        bundles = (
            list(truth)
            if candidate_bundles_by_bidder is None
            else list(candidate_bundles_by_bidder[bidder_id])
        )
        budget = (
            None
            if disclosed_budgets is None
            else float(disclosed_budgets[bidder_id])
        )
        for bundle in sorted(bundles, key=lambda b: (len(b), sorted(b))):
            true_value = float(truth[bundle])
            raw = true_calibration.invert(true_value, len(bundle))
            if noise_scale:
                raw *= 1.0 + rng.uniform(-noise_scale, noise_scale)
            observations.append(
                PvObservation(
                    domain=domain,
                    seed=scenario_seed,
                    bidder_id=bidder_id,
                    bundle=frozenset(bundle),
                    raw_value=max(0.0, raw),
                    true_value=true_value,
                    disclosed_budget=budget,
                )
            )
    return observations


def synthetic_artefact(
    domain: str,
    *,
    seed: int = 0,
    true_calibration: ValueCalibration,
    noise_scale: float = 0.0,
    noise_seed: int = 0,
    max_bundle_size: int | None = None,
) -> dict[str, Any]:
    """Build a frozen-artefact payload from a known calibration, offline.

    Same schema as a real prepared artefact, but the "predictions" are derived
    from the hidden truth by inverting ``true_calibration`` instead of being
    produced by a model. Used to test the fitter and to rehearse the fitting
    pipeline without spending a single API call. ``models.proxy.provider`` is
    set to ``"synthetic"`` so such an artefact can never be mistaken for a
    measurement of a real estimator.
    """
    scenario = build_benchmark_scenario(domain, seed=seed)
    candidates = {
        bidder_id: [
            bundle
            for bundle in scenario.instance.valuations[bidder_id]
            if max_bundle_size is None or len(bundle) <= max_bundle_size
        ]
        for bidder_id in scenario.instance.bidder_ids
    }
    budgets = {
        bidder_id: max(
            scenario.instance.valuations[bidder_id].values(), default=0.0
        )
        for bidder_id in scenario.instance.bidder_ids
    }
    observations = synthesize_observations(
        scenario,
        true_calibration=true_calibration,
        candidate_bundles_by_bidder=candidates,
        noise_scale=noise_scale,
        seed=noise_seed,
        disclosed_budgets=budgets,
    )
    by_bidder: dict[str, list[PvObservation]] = {}
    for observation in observations:
        by_bidder.setdefault(observation.bidder_id, []).append(observation)

    return {
        "format": PV_CALIBRATION_BENCHMARK_FORMAT,
        "version": PV_CALIBRATION_BENCHMARK_VERSION,
        "domain": domain,
        "seed": seed,
        "scenario_name": scenario.name,
        "items": list(scenario.instance.items),
        "bidder_ids": list(by_bidder),
        "skipped_bidder_ids": [],
        "prompt_versions": {},
        "models": {
            "person": {"provider": "synthetic", "model": None, "temperature": None},
            "proxy": {"provider": "synthetic", "model": None, "temperature": None},
            "verifier": {"provider": None, "model": None, "temperature": None},
        },
        "generation_settings": {
            "synthetic": True,
            "true_calibration": true_calibration.to_dict(),
            "noise_scale": noise_scale,
            "max_bundle_size": max_bundle_size,
        },
        "call_metadata": {},
        "bidders": {
            bidder_id: {
                "nl_question": "synthetic",
                "nl_answer": "synthetic",
                "interest_map": None,
                "disclosed_budget": budgets[bidder_id],
                "candidate_bundles": [
                    _bundle_list(o.bundle) for o in bidder_observations
                ],
                "raw_provisional_values": [
                    {"bundle": _bundle_list(o.bundle), "value": o.raw_value}
                    for o in bidder_observations
                ],
                "hidden_true_values": [
                    {"bundle": _bundle_list(o.bundle), "value": o.true_value}
                    for o in bidder_observations
                ],
                "diagnostics": {},
            }
            for bidder_id, bidder_observations in by_bidder.items()
        },
    }


# ---------------------------------------------------------------------------
# Row builders for CSV output
# ---------------------------------------------------------------------------

def observation_rows(
    observations: Sequence[PvObservation],
    calibration: ValueCalibration,
) -> list[dict[str, Any]]:
    """One row per (bidder, bundle), raw and calibrated side by side."""
    raw = ValueCalibration(family="none")
    rows: list[dict[str, Any]] = []
    for observation in observations:
        raw_prediction = predict(observation, raw)
        calibrated = predict(observation, calibration)
        rows.append(
            {
                "domain": observation.domain,
                "seed": observation.seed,
                "bidder_id": observation.bidder_id,
                "bundle": bundle_key(observation.bundle),
                "bundle_size": observation.bundle_size,
                "true_value": observation.true_value,
                "disclosed_budget": (
                    "" if observation.disclosed_budget is None
                    else observation.disclosed_budget
                ),
                "raw_value": raw_prediction,
                "calibrated_value": calibrated,
                "raw_signed_error": raw_prediction - observation.true_value,
                "calibrated_signed_error": calibrated - observation.true_value,
                "raw_abs_error": abs(raw_prediction - observation.true_value),
                "calibrated_abs_error": abs(calibrated - observation.true_value),
                "true_over_raw": (
                    (observation.true_value + LOG_ERROR_FLOOR)
                    / (raw_prediction + LOG_ERROR_FLOOR)
                ),
            }
        )
    return rows


def metrics_row(
    scope: str,
    label: str,
    variant: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"scope": scope, "label": label, "variant": variant}
    for key, value in metrics.items():
        row[key] = "" if value is None else value
    return row


METRIC_FIELDS: tuple[str, ...] = (
    "scope",
    "label",
    "variant",
    "n",
    "mae",
    "rmse",
    "budget_normalized_mae",
    "robust_log_error",
    "signed_bias",
    "budget_normalized_signed_bias",
    "median_true_over_predicted",
    "spearman",
    "mean_bidder_spearman",
    "top_1_recall",
    "top_3_recall",
    "top_5_recall",
)
