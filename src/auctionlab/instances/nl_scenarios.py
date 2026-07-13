from __future__ import annotations

from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_types import NaturalLanguageAuctionScenario

__all__ = [
    "NaturalLanguageAuctionScenario",
    "curated_natural_language_scenarios",
    "legacy_natural_language_scenarios",
]


def curated_natural_language_scenarios() -> list[NaturalLanguageAuctionScenario]:
    """Return the canonical set of generated PC-build structured scenarios.

    These replace the old hand-written explicit/implicit seeds. See
    :func:`legacy_natural_language_scenarios` for the original home-office
    and home-studio scenarios (kept for reproducibility).
    """
    from auctionlab.instances.structured import make_pc_build_scenario  # lazy to avoid circular

    return [
        make_pc_build_scenario(4, 4, seed=0),
        make_pc_build_scenario(6, 6, seed=0),
        make_pc_build_scenario(8, 8, seed=0),
        make_pc_build_scenario(10, 10, seed=0),
    ]


def legacy_natural_language_scenarios() -> list[NaturalLanguageAuctionScenario]:
    return [
        NaturalLanguageAuctionScenario(
            name="home_office_5x5",
            seed_type="implicit",
            instance=AuctionInstance(
                items=["MONITOR", "MKEYSET", "WEBCAM", "HEADSET", "RINGLIGHT"],
                bidder_ids=[
                    "developer",
                    "streamer",
                    "manager",
                    "student",
                    "reseller",
                ],
                valuations={
                    "developer": {
                        frozenset({"MONITOR"}): 450.0,
                        frozenset({"MKEYSET"}): 180.0,
                        frozenset({"WEBCAM"}): 70.0,
                        frozenset({"HEADSET"}): 140.0,
                        frozenset({"RINGLIGHT"}): 30.0,
                        frozenset({"MONITOR", "MKEYSET"}): 690.0,
                        frozenset({"MONITOR", "HEADSET"}): 570.0,
                        frozenset({"MONITOR", "MKEYSET", "HEADSET"}): 800.0,
                        frozenset({"MONITOR", "MKEYSET", "HEADSET", "WEBCAM"}): 850.0,
                    },
                    "streamer": {
                        frozenset({"MONITOR"}): 190.0,
                        frozenset({"MKEYSET"}): 70.0,
                        frozenset({"WEBCAM"}): 340.0,
                        frozenset({"HEADSET"}): 180.0,
                        frozenset({"RINGLIGHT"}): 260.0,
                        frozenset({"WEBCAM", "RINGLIGHT"}): 700.0,
                        frozenset({"WEBCAM", "HEADSET"}): 540.0,
                        frozenset({"HEADSET", "RINGLIGHT"}): 400.0,
                        frozenset({"WEBCAM", "RINGLIGHT", "HEADSET"}): 860.0,
                        frozenset({"WEBCAM", "RINGLIGHT", "MONITOR"}): 870.0,
                        frozenset({"WEBCAM", "RINGLIGHT", "HEADSET", "MONITOR"}): 1040.0,
                    },
                    "manager": {
                        frozenset({"MONITOR"}): 300.0,
                        frozenset({"MKEYSET"}): 100.0,
                        frozenset({"WEBCAM"}): 280.0,
                        frozenset({"HEADSET"}): 310.0,
                        frozenset({"RINGLIGHT"}): 140.0,
                        frozenset({"HEADSET", "WEBCAM"}): 630.0,
                        frozenset({"HEADSET", "RINGLIGHT"}): 420.0,
                        frozenset({"HEADSET", "WEBCAM", "RINGLIGHT"}): 750.0,
                        frozenset({"MONITOR", "HEADSET", "WEBCAM"}): 880.0,
                        frozenset({"MONITOR", "HEADSET", "WEBCAM", "RINGLIGHT"}): 980.0,
                    },
                    "student": {
                        frozenset({"MONITOR"}): 290.0,
                        frozenset({"MKEYSET"}): 140.0,
                        frozenset({"WEBCAM"}): 90.0,
                        frozenset({"HEADSET"}): 110.0,
                        frozenset({"RINGLIGHT"}): 50.0,
                        frozenset({"MONITOR", "MKEYSET"}): 410.0,
                        frozenset({"MONITOR", "HEADSET"}): 375.0,
                        frozenset({"MONITOR", "MKEYSET", "HEADSET"}): 510.0,
                    },
                    "reseller": {
                        frozenset({"MONITOR"}): 280.0,
                        frozenset({"MKEYSET"}): 120.0,
                        frozenset({"WEBCAM"}): 190.0,
                        frozenset({"HEADSET"}): 170.0,
                        frozenset({"RINGLIGHT"}): 90.0,
                        frozenset({"MONITOR", "MKEYSET"}): 420.0,
                        frozenset({"WEBCAM", "HEADSET", "RINGLIGHT"}): 480.0,
                        frozenset({"MONITOR", "MKEYSET", "WEBCAM", "HEADSET", "RINGLIGHT"}): 920.0,
                    },
                },
            ),
            scenario_description=(
                "A local co-working space is auctioning five items from its "
                "equipment surplus: a 4K monitor, a mechanical keyboard and "
                "mouse set, a 4K webcam, a professional noise-cancelling "
                "headset, and a professional LED ring light. All items are "
                "in excellent working condition."
            ),
            item_descriptions={
                "MONITOR": (
                    "LG 27UK850-W 27-inch 4K UHD IPS Monitor (2022): "
                    "3840x2160 resolution with HDR10 support and 95% DCI-P3 "
                    "wide colour gamut. USB-C with 60W power delivery, two "
                    "HDMI 2.0 ports, AMD FreeSync. Height-adjustable stand, "
                    "VESA 100x100mm compatible. Response time 5ms, 60Hz."
                ),
                "MKEYSET": (
                    "Keychron K2 Mechanical Keyboard + Logitech MX Master 3S "
                    "Mouse Set: Keychron K2 75% wireless keyboard with Gateron "
                    "G Pro Red switches, hot-swappable PCB, per-key RGB "
                    "backlight, Bluetooth 5.1 and 2.4GHz. Logitech MX Master "
                    "3S mouse with 8000 DPI MagSpeed scroll, 7 programmable "
                    "buttons, Bolt USB receiver."
                ),
                "WEBCAM": (
                    "Logitech Brio 4K Ultra HD Webcam (2023): 4K at 30fps or "
                    "1080p at 60fps video. 90-degree adjustable field of view, "
                    "built-in dual noise-cancelling microphones, HDR support, "
                    "RightLight 4 for low-light correction. USB-C connection. "
                    "Windows Hello facial recognition support."
                ),
                "HEADSET": (
                    "Jabra Evolve2 75 Wireless Noise-Cancelling Headset (2023): "
                    "Hybrid ANC with 8-microphone array for clear call audio. "
                    "Bluetooth 5.0, 36-hour battery with ANC on. Boom arm "
                    "microphone. Certified for Microsoft Teams and UC platforms. "
                    "Charging stand included. Leak-resistant speakers."
                ),
                "RINGLIGHT": (
                    "Elgato Key Light Professional LED Panel (2022): 2800 lumens, "
                    "40W. Colour temperature adjustable 2500–6500K, brightness "
                    "0–100% via app. 13.2-inch ultra-thin panel on adjustable "
                    "arm mount. Wi-Fi connected, controlled via Elgato Control "
                    "Center or Stream Deck. Flicker-free at all brightness levels."
                ),
            },
            person_seeds={
                "developer": (
                    "The developer works remotely full-time and needs a "
                    "productive desk setup. The large 4K monitor and the "
                    "keyboard-and-mouse set are the two most important items "
                    "and are strongly complementary: a large display is nearly "
                    "useless without good input devices, and together they form "
                    "a complete coding workstation worth substantially more than "
                    "either item alone. A good noise-cancelling headset adds "
                    "meaningful focus and is useful for team calls. The webcam "
                    "is a minor convenience for video meetings. The ring light "
                    "has almost no value for their work."
                ),
                "streamer": (
                    "The streamer runs a live-streaming channel and needs "
                    "high production quality on camera. The 4K webcam and the "
                    "professional ring light are strongly complementary: good "
                    "lighting is useless without a camera to capture it, and "
                    "the camera is far less effective in poor light, so together "
                    "they form a streaming visual setup worth substantially more "
                    "than either alone. The noise-cancelling headset is important "
                    "for monitoring audio and communicating with the audience. "
                    "The monitor is useful for watching stream metrics and chat. "
                    "The keyboard set is a convenience but not central."
                ),
                "manager": (
                    "The manager spends most of the working day on video calls "
                    "and virtual presentations. The noise-cancelling headset is "
                    "their highest-priority item because clear audio is critical "
                    "on calls. Paired with the 4K webcam, the headset and webcam "
                    "together form their core meeting setup: professional video "
                    "and audio are complementary because a meeting with bad audio "
                    "or bad video both undermine communication. Good lighting "
                    "further improves on-camera presence. A large monitor helps "
                    "with multitasking between applications during meetings. The "
                    "keyboard set is a basic secondary need."
                ),
                "student": (
                    "The student needs a practical study environment on a limited "
                    "income. The large 4K monitor is the highest-priority item "
                    "because a big screen dramatically improves reading and writing "
                    "long documents. Paired with the keyboard set, it forms a "
                    "solid home study station. The webcam is useful for attending "
                    "online lectures and group video calls. The headset helps "
                    "with focus and blocking distractions. The ring light is a "
                    "nice extra but not essential."
                ),
                "reseller": (
                    "The reseller sees resale potential across all five items. "
                    "The 4K monitor commands the highest individual margin. A "
                    "webcam, headset, and ring light bundled together sell well "
                    "as a complete streaming or meetings kit — the combination "
                    "is more attractive to buyers than the parts individually. "
                    "The keyboard set has moderate margins. The complete "
                    "five-item home-office package is the best-case scenario "
                    "but requires finding one premium buyer."
                ),
            },
            candidate_bundles_by_bidder={
                "developer": [
                    frozenset({"MONITOR"}),
                    frozenset({"MKEYSET"}),
                    frozenset({"HEADSET"}),
                    frozenset({"WEBCAM"}),
                    frozenset({"MONITOR", "MKEYSET"}),
                    frozenset({"MONITOR", "HEADSET"}),
                    frozenset({"MONITOR", "MKEYSET", "HEADSET"}),
                ],
                "streamer": [
                    frozenset({"WEBCAM"}),
                    frozenset({"RINGLIGHT"}),
                    frozenset({"HEADSET"}),
                    frozenset({"MONITOR"}),
                    frozenset({"WEBCAM", "RINGLIGHT"}),
                    frozenset({"WEBCAM", "HEADSET"}),
                    frozenset({"WEBCAM", "RINGLIGHT", "HEADSET"}),
                    frozenset({"WEBCAM", "RINGLIGHT", "MONITOR"}),
                ],
                "manager": [
                    frozenset({"HEADSET"}),
                    frozenset({"WEBCAM"}),
                    frozenset({"RINGLIGHT"}),
                    frozenset({"MONITOR"}),
                    frozenset({"HEADSET", "WEBCAM"}),
                    frozenset({"HEADSET", "RINGLIGHT"}),
                    frozenset({"HEADSET", "WEBCAM", "RINGLIGHT"}),
                    frozenset({"MONITOR", "HEADSET", "WEBCAM"}),
                ],
                "student": [
                    frozenset({"MONITOR"}),
                    frozenset({"MKEYSET"}),
                    frozenset({"WEBCAM"}),
                    frozenset({"HEADSET"}),
                    frozenset({"MONITOR", "MKEYSET"}),
                    frozenset({"MONITOR", "HEADSET"}),
                ],
                "reseller": [
                    frozenset({"MONITOR"}),
                    frozenset({"MKEYSET"}),
                    frozenset({"WEBCAM"}),
                    frozenset({"HEADSET"}),
                    frozenset({"RINGLIGHT"}),
                    frozenset({"MONITOR", "MKEYSET"}),
                    frozenset({"WEBCAM", "HEADSET", "RINGLIGHT"}),
                ],
            },
        ),
        NaturalLanguageAuctionScenario(
            name="home_studio_6x6",
            seed_type="explicit",
            instance=AuctionInstance(
                items=[
                    "INTERFACE",
                    "MICROPHONE",
                    "MONITORS",
                    "MIDI_KEYS",
                    "HEADPHONES",
                    "PANELS",
                ],
                bidder_ids=[
                    "singer_songwriter",
                    "music_producer",
                    "podcaster",
                    "mixing_engineer",
                    "live_performer",
                    "studio_reseller",
                ],
                valuations={
                    "singer_songwriter": {
                        frozenset({"MICROPHONE"}): 200.0,
                        frozenset({"INTERFACE"}): 50.0,
                        frozenset({"PANELS"}): 180.0,
                        frozenset({"HEADPHONES"}): 100.0,
                        frozenset({"MIDI_KEYS"}): 80.0,
                        frozenset({"MICROPHONE", "INTERFACE"}): 500.0,
                        frozenset({"MICROPHONE", "INTERFACE", "PANELS"}): 750.0,
                        frozenset({"MICROPHONE", "INTERFACE", "HEADPHONES"}): 620.0,
                        frozenset({
                            "MICROPHONE", "INTERFACE", "PANELS", "HEADPHONES"
                        }): 880.0,
                        frozenset({
                            "MICROPHONE", "INTERFACE", "PANELS",
                            "HEADPHONES", "MIDI_KEYS",
                        }): 950.0,
                    },
                    "music_producer": {
                        frozenset({"MIDI_KEYS"}): 180.0,
                        frozenset({"INTERFACE"}): 200.0,
                        frozenset({"MONITORS"}): 400.0,
                        frozenset({"HEADPHONES"}): 120.0,
                        frozenset({"MIDI_KEYS", "INTERFACE"}): 450.0,
                        frozenset({"MIDI_KEYS", "INTERFACE", "HEADPHONES"}): 600.0,
                        frozenset({"MIDI_KEYS", "INTERFACE", "MONITORS"}): 900.0,
                        frozenset({
                            "MIDI_KEYS", "INTERFACE", "MONITORS", "HEADPHONES"
                        }): 950.0,
                    },
                    "podcaster": {
                        frozenset({"MICROPHONE"}): 250.0,
                        frozenset({"INTERFACE"}): 50.0,
                        frozenset({"PANELS"}): 220.0,
                        frozenset({"HEADPHONES"}): 100.0,
                        frozenset({"MICROPHONE", "INTERFACE"}): 480.0,
                        frozenset({"MICROPHONE", "INTERFACE", "PANELS"}): 780.0,
                        frozenset({"MICROPHONE", "INTERFACE", "HEADPHONES"}): 580.0,
                        frozenset({
                            "MICROPHONE", "INTERFACE", "PANELS", "HEADPHONES"
                        }): 880.0,
                    },
                    "mixing_engineer": {
                        frozenset({"MONITORS"}): 500.0,
                        frozenset({"INTERFACE"}): 300.0,
                        frozenset({"HEADPHONES"}): 200.0,
                        frozenset({"PANELS"}): 400.0,
                        frozenset({"MONITORS", "INTERFACE"}): 950.0,
                        frozenset({"MONITORS", "INTERFACE", "HEADPHONES"}): 1100.0,
                        frozenset({"MONITORS", "INTERFACE", "PANELS"}): 1450.0,
                        frozenset({
                            "MONITORS", "INTERFACE", "HEADPHONES", "PANELS"
                        }): 1700.0,
                    },
                    "live_performer": {
                        frozenset({"MIDI_KEYS"}): 250.0,
                        frozenset({"INTERFACE"}): 200.0,
                        frozenset({"MONITORS"}): 300.0,
                        frozenset({"MIDI_KEYS", "INTERFACE"}): 550.0,
                        frozenset({"MIDI_KEYS", "INTERFACE", "MONITORS"}): 800.0,
                    },
                    "studio_reseller": {
                        frozenset({"INTERFACE"}): 300.0,
                        frozenset({"MICROPHONE"}): 230.0,
                        frozenset({"HEADPHONES"}): 120.0,
                        frozenset({"MONITORS"}): 420.0,
                        frozenset({"PANELS"}): 160.0,
                        frozenset({"MIDI_KEYS"}): 150.0,
                        frozenset({"INTERFACE", "MICROPHONE"}): 580.0,
                        frozenset({"INTERFACE", "MICROPHONE", "HEADPHONES"}): 720.0,
                        frozenset({"MONITORS", "PANELS"}): 650.0,
                        frozenset({
                            "INTERFACE", "MICROPHONE", "HEADPHONES",
                            "MONITORS", "PANELS",
                        }): 1450.0,
                    },
                },
            ),
            scenario_description=(
                "A home-studio equipment auction among six musicians and audio "
                "professionals. Six items are available: an audio interface "
                "(INTERFACE), a condenser microphone (MICROPHONE), a pair of "
                "studio monitor speakers (MONITORS), a MIDI keyboard controller "
                "(MIDI_KEYS), studio headphones (HEADPHONES), and acoustic foam "
                "panels (PANELS). Each bidder has a specific professional role "
                "that determines which items and bundles are most valuable to them."
            ),
            item_descriptions={
                "INTERFACE": (
                    "USB audio interface for connecting microphones and instruments "
                    "to a computer. Required to use professional microphones and "
                    "essential for any recording or live-performance rig."
                ),
                "MICROPHONE": (
                    "Large-diaphragm condenser microphone for vocal and instrument "
                    "recording. Requires an audio interface to function properly."
                ),
                "MONITORS": (
                    "Pair of studio reference monitor speakers for accurate audio "
                    "playback. Essential for mixing and mastering; also useful as "
                    "stage monitors for live performance."
                ),
                "MIDI_KEYS": (
                    "49-key MIDI keyboard controller for programming beats, "
                    "melodies, and harmonies. Works with any DAW via USB."
                ),
                "HEADPHONES": (
                    "Closed-back studio headphones for monitoring, mixing checks, "
                    "and silent practice."
                ),
                "PANELS": (
                    "Set of acoustic foam panels for dampening room reflections "
                    "and improving recorded sound quality."
                ),
            },
            person_seeds={
                "singer_songwriter": (
                    "Maya is a bedroom pop singer-songwriter who records her own "
                    "vocals and guitar at home. She is focused on acquiring the "
                    "MICROPHONE for her recordings, placing a high value on "
                    "capturing clean vocal takes. The dollar value for the "
                    "MICROPHONE she is willing to pay is estimated at $200.\n\n"
                    "1. Single Item Bundles:\n"
                    "- MICROPHONE - $200\n"
                    "- INTERFACE - $50\n"
                    "- PANELS - $180\n"
                    "- HEADPHONES - $100\n"
                    "- MIDI_KEYS - $80\n\n"
                    "2. Complementary Bundles:\n"
                    "- Home-recording signal chain:\n"
                    "  - MICROPHONE - $200\n"
                    "  - INTERFACE - $50\n"
                    "  - Total Value: $500\n"
                    "- Signal chain with acoustic treatment:\n"
                    "  - MICROPHONE, INTERFACE - $500 (as a pair)\n"
                    "  - PANELS - $180\n"
                    "  - Total Value: $750\n"
                    "- Signal chain with mixing headphones:\n"
                    "  - MICROPHONE, INTERFACE - $500 (as a pair)\n"
                    "  - HEADPHONES - $100\n"
                    "  - Total Value: $620\n"
                    "- Full recording setup:\n"
                    "  - MICROPHONE, INTERFACE, PANELS - $750 (as a triple)\n"
                    "  - HEADPHONES - $100\n"
                    "  - Total Value: $880\n"
                    "- Full setup with melody sketching:\n"
                    "  - MICROPHONE, INTERFACE, PANELS, HEADPHONES - $880\n"
                    "  - MIDI_KEYS - $80\n"
                    "  - Total Value: $950\n\n"
                    "Maya values MICROPHONE and INTERFACE as a strongly "
                    "complementary pair: bought together they are worth 100% "
                    "more than the sum of their individual prices, since "
                    "neither is much use to her without the other. Adding "
                    "PANELS or HEADPHONES to that core pair still carries a "
                    "complementary premium of roughly 70-80% over the additive "
                    "sum of all items involved, reflecting how much a properly "
                    "treated, monitored vocal booth improves her recordings. "
                    "MIDI_KEYS is a substitute-adjacent add-on rather than a "
                    "core complement: its marginal contribution to her full "
                    "setup is only about $70, a 12.5% discount on its $80 "
                    "standalone value, since melody sketching is secondary to "
                    "her vocal-focused workflow. Maya has no use for MONITORS "
                    "at all — she always mixes on headphones — so any bundle "
                    "built around MONITORS without her core recording gear is "
                    "worth $0 to her, a full substitution discount."
                ),
                "music_producer": (
                    "Jordan runs a hip-hop and electronic music production "
                    "studio. He is focused on acquiring the INTERFACE as the "
                    "centerpiece of his production workstation, willing to pay "
                    "an estimated $200 for it.\n\n"
                    "1. Single Item Bundles:\n"
                    "- INTERFACE - $200\n"
                    "- MIDI_KEYS - $180\n"
                    "- MONITORS - $400\n"
                    "- HEADPHONES - $120\n\n"
                    "2. Complementary Bundles:\n"
                    "- Production controller pair:\n"
                    "  - MIDI_KEYS - $180\n"
                    "  - INTERFACE - $200\n"
                    "  - Total Value: $450\n"
                    "- Full production workstation:\n"
                    "  - MIDI_KEYS, INTERFACE - $450 (as a pair)\n"
                    "  - MONITORS - $400\n"
                    "  - Total Value: $900\n"
                    "- Workstation with reference headphones:\n"
                    "  - MIDI_KEYS, INTERFACE, MONITORS - $900 (as a triple)\n"
                    "  - HEADPHONES - $120\n"
                    "  - Total Value: $950\n\n"
                    "Jordan values MIDI_KEYS and INTERFACE as a complementary "
                    "pair worth about 18% more together than apart, since a "
                    "controller without an interface to route audio through is "
                    "of limited use. Adding MONITORS to that pair is worth even "
                    "more than its $400 standalone price — about a 12.5% "
                    "premium — because accurate monitoring is what makes the "
                    "rest of the workstation usable for finished mixes. "
                    "HEADPHONES, by contrast, shows clear bulk-discount "
                    "behavior: once he already has monitors, headphones add "
                    "only about $50 of marginal value versus their $120 "
                    "standalone price, a roughly 58% discount, since they "
                    "become a redundant secondary reference rather than an "
                    "essential tool. Jordan does not record live vocals or "
                    "podcasts, so MICROPHONE is worth $0 to him, and his "
                    "studio is already acoustically treated, so PANELS is "
                    "also worth $0 in any bundle without his core production "
                    "gear — both are pure substitution discounts of 100%."
                ),
                "podcaster": (
                    "Sam records a weekly interview podcast and is focused on "
                    "acquiring the MICROPHONE, which she considers the "
                    "cornerstone of her setup. The dollar value she is willing "
                    "to pay for the MICROPHONE is estimated at $250.\n\n"
                    "1. Single Item Bundles:\n"
                    "- MICROPHONE - $250\n"
                    "- PANELS - $220\n"
                    "- INTERFACE - $50\n"
                    "- HEADPHONES - $100\n\n"
                    "2. Complementary Bundles:\n"
                    "- Microphone signal chain:\n"
                    "  - MICROPHONE - $250\n"
                    "  - INTERFACE - $50\n"
                    "  - Total Value: $480\n"
                    "- Signal chain with room treatment:\n"
                    "  - MICROPHONE, INTERFACE - $480 (as a pair)\n"
                    "  - PANELS - $220\n"
                    "  - Total Value: $780\n"
                    "- Near-complete podcasting setup:\n"
                    "  - MICROPHONE, INTERFACE, PANELS - $780 (as a triple)\n"
                    "  - HEADPHONES - $100\n"
                    "  - Total Value: $880\n\n"
                    "Sam values MICROPHONE and INTERFACE as a strongly "
                    "complementary pair, worth about 60% more together than "
                    "the sum of their individual prices, since the microphone "
                    "is unusable for digital recording without the interface. "
                    "PANELS carries an even larger complementary premium when "
                    "added to that pair — its marginal contribution is about "
                    "$300, roughly 36% above its $220 standalone value, "
                    "because reducing room echo meaningfully improves the "
                    "perceived quality of every recorded episode. HEADPHONES "
                    "is a flat, non-discounted add-on: its marginal "
                    "contribution to the full bundle exactly matches its $100 "
                    "standalone value, since it serves a separable monitoring "
                    "function rather than reinforcing the others. Sam has no "
                    "use for MONITORS or MIDI_KEYS, which are irrelevant to "
                    "podcasting, so she assigns $0 to any bundle built around "
                    "those items alone — a full substitution discount."
                ),
                "mixing_engineer": (
                    "Alex is a professional mixing engineer who works on "
                    "commercial tracks. He is focused on acquiring MONITORS, "
                    "his most critical item, placing a high value on accurate "
                    "studio reference playback. The dollar value he is willing "
                    "to pay for MONITORS is estimated at $500.\n\n"
                    "1. Single Item Bundles:\n"
                    "- MONITORS - $500\n"
                    "- PANELS - $400\n"
                    "- INTERFACE - $300\n"
                    "- HEADPHONES - $200\n\n"
                    "2. Complementary Bundles:\n"
                    "- Core monitoring chain:\n"
                    "  - MONITORS - $500\n"
                    "  - INTERFACE - $300\n"
                    "  - Total Value: $950\n"
                    "- Monitoring chain with headphone reference:\n"
                    "  - MONITORS, INTERFACE - $950 (as a pair)\n"
                    "  - HEADPHONES - $200\n"
                    "  - Total Value: $1,100\n"
                    "- Minimum viable mix room:\n"
                    "  - MONITORS, INTERFACE - $950 (as a pair)\n"
                    "  - PANELS - $400\n"
                    "  - Total Value: $1,450\n"
                    "- Complete professional setup:\n"
                    "  - MONITORS, INTERFACE, PANELS - $1,450 (as a triple)\n"
                    "  - HEADPHONES - $200\n"
                    "  - Total Value: $1,700\n\n"
                    "Alex values MONITORS and INTERFACE as the indispensable "
                    "core of his rig, worth about 19% more as a pair than the "
                    "sum of their prices. PANELS carries a substantial "
                    "complementary premium on top of that pair — roughly 25% "
                    "above its $400 standalone value — because accurate room "
                    "acoustics are what let the monitors be trusted for "
                    "professional mix decisions. HEADPHONES shows a mild bulk "
                    "discount when added to the core pair alone (about a 25% "
                    "reduction on its marginal contribution), but once PANELS "
                    "is already present, HEADPHONES regains a premium of about "
                    "25% over its standalone value, since the full four-item "
                    "kit functions as a genuinely integrated professional "
                    "setup rather than a collection of redundant tools. Alex "
                    "has no interest in MICROPHONE (he uses studio mics not "
                    "listed here) or MIDI_KEYS (he is not a performer), so "
                    "those items are worth $0 to him in any bundle that does "
                    "not include his core monitoring gear."
                ),
                "live_performer": (
                    "Chris performs live electronic and synth sets. He is "
                    "focused on acquiring MIDI_KEYS, his essential performance "
                    "controller, and is willing to pay an estimated $250 for "
                    "it.\n\n"
                    "1. Single Item Bundles:\n"
                    "- MIDI_KEYS - $250\n"
                    "- INTERFACE - $200\n"
                    "- MONITORS - $300\n\n"
                    "2. Complementary Bundles:\n"
                    "- Compact live rig:\n"
                    "  - MIDI_KEYS - $250\n"
                    "  - INTERFACE - $200\n"
                    "  - Total Value: $550\n"
                    "- Ideal portable live setup:\n"
                    "  - MIDI_KEYS, INTERFACE - $550 (as a pair)\n"
                    "  - MONITORS - $300\n"
                    "  - Total Value: $800\n\n"
                    "Chris values MIDI_KEYS and INTERFACE as a complementary "
                    "pair worth about 22% more together than apart, since "
                    "routing the controller's audio on stage requires the "
                    "interface. MONITORS shows mild bulk-discount behavior "
                    "when added to that pair: its marginal contribution is "
                    "about $250, a 17% discount on its $300 standalone value, "
                    "because Chris primarily monitors through in-ear gear and "
                    "treats stage monitors as a secondary convenience rather "
                    "than essential. Chris never records in a studio, so "
                    "MICROPHONE, PANELS, and HEADPHONES are worth $0 to him in "
                    "any bundle built around those items alone — a full "
                    "substitution discount, since none of them serve his live "
                    "performance workflow."
                ),
                "studio_reseller": (
                    "Dana buys and resells home-studio gear bundles to music "
                    "students. Unlike the other bidders, she has no single "
                    "indispensable item — her valuations are driven by resale "
                    "market prices across the whole catalog rather than "
                    "personal workflow complementarity.\n\n"
                    "1. Single Item Bundles:\n"
                    "- INTERFACE - $300\n"
                    "- MONITORS - $420\n"
                    "- MICROPHONE - $230\n"
                    "- PANELS - $160\n"
                    "- MIDI_KEYS - $150\n"
                    "- HEADPHONES - $120\n\n"
                    "2. Complementary Bundles:\n"
                    "- 'Beginner recording kit' (resale package):\n"
                    "  - INTERFACE - $300\n"
                    "  - MICROPHONE - $230\n"
                    "  - Total Value: $580\n"
                    "- Beginner kit with headphones:\n"
                    "  - INTERFACE, MICROPHONE - $580 (as a pair)\n"
                    "  - HEADPHONES - $120\n"
                    "  - Total Value: $720\n"
                    "- 'Room treatment and monitoring upgrade':\n"
                    "  - MONITORS - $420\n"
                    "  - PANELS - $160\n"
                    "  - Total Value: $650\n"
                    "- Near-complete studio kit:\n"
                    "  - INTERFACE, MICROPHONE, HEADPHONES - $720 (as a triple)\n"
                    "  - MONITORS, PANELS - $650 (as a pair)\n"
                    "  - Total Value: $1,450\n\n"
                    "Dana applies a consistent, modest bundling premium of "
                    "roughly 9-18% above the sum of component resale prices "
                    "across all of her packaged kits, reflecting that a "
                    "pre-assembled bundle sells faster and at a better margin "
                    "than loose individual items, not that she personally "
                    "needs the items together. This is a much weaker effect "
                    "than the 50-100% complementary premiums the other "
                    "bidders place on their core workflow pairs: Dana never "
                    "discounts an item to $0, since every item has resale "
                    "value on its own, but she also never prioritizes one "
                    "item enough to justify steep complementary or "
                    "substitution discounts. She can profit from any subset "
                    "of the catalog."
                ),
            },
            candidate_bundles_by_bidder={
                "singer_songwriter": [
                    frozenset({"MICROPHONE"}),
                    frozenset({"INTERFACE"}),
                    frozenset({"PANELS"}),
                    frozenset({"HEADPHONES"}),
                    frozenset({"MIDI_KEYS"}),
                    frozenset({"MICROPHONE", "INTERFACE"}),
                    frozenset({"MICROPHONE", "INTERFACE", "PANELS"}),
                    frozenset({"MICROPHONE", "INTERFACE", "HEADPHONES"}),
                    frozenset({
                        "MICROPHONE", "INTERFACE", "PANELS", "HEADPHONES"
                    }),
                    frozenset({
                        "MICROPHONE", "INTERFACE", "PANELS",
                        "HEADPHONES", "MIDI_KEYS",
                    }),
                ],
                "music_producer": [
                    frozenset({"MIDI_KEYS"}),
                    frozenset({"INTERFACE"}),
                    frozenset({"MONITORS"}),
                    frozenset({"HEADPHONES"}),
                    frozenset({"MIDI_KEYS", "INTERFACE"}),
                    frozenset({"MIDI_KEYS", "INTERFACE", "HEADPHONES"}),
                    frozenset({"MIDI_KEYS", "INTERFACE", "MONITORS"}),
                    frozenset({
                        "MIDI_KEYS", "INTERFACE", "MONITORS", "HEADPHONES"
                    }),
                ],
                "podcaster": [
                    frozenset({"MICROPHONE"}),
                    frozenset({"INTERFACE"}),
                    frozenset({"PANELS"}),
                    frozenset({"HEADPHONES"}),
                    frozenset({"MICROPHONE", "INTERFACE"}),
                    frozenset({"MICROPHONE", "INTERFACE", "PANELS"}),
                    frozenset({"MICROPHONE", "INTERFACE", "HEADPHONES"}),
                    frozenset({
                        "MICROPHONE", "INTERFACE", "PANELS", "HEADPHONES"
                    }),
                ],
                "mixing_engineer": [
                    frozenset({"MONITORS"}),
                    frozenset({"INTERFACE"}),
                    frozenset({"HEADPHONES"}),
                    frozenset({"PANELS"}),
                    frozenset({"MONITORS", "INTERFACE"}),
                    frozenset({"MONITORS", "INTERFACE", "HEADPHONES"}),
                    frozenset({"MONITORS", "INTERFACE", "PANELS"}),
                    frozenset({
                        "MONITORS", "INTERFACE", "HEADPHONES", "PANELS"
                    }),
                ],
                "live_performer": [
                    frozenset({"MIDI_KEYS"}),
                    frozenset({"INTERFACE"}),
                    frozenset({"MONITORS"}),
                    frozenset({"MIDI_KEYS", "INTERFACE"}),
                    frozenset({"MIDI_KEYS", "INTERFACE", "MONITORS"}),
                ],
                "studio_reseller": [
                    frozenset({"INTERFACE"}),
                    frozenset({"MICROPHONE"}),
                    frozenset({"HEADPHONES"}),
                    frozenset({"MONITORS"}),
                    frozenset({"PANELS"}),
                    frozenset({"MIDI_KEYS"}),
                    frozenset({"INTERFACE", "MICROPHONE"}),
                    frozenset({"INTERFACE", "MICROPHONE", "HEADPHONES"}),
                    frozenset({"MONITORS", "PANELS"}),
                    frozenset({
                        "INTERFACE", "MICROPHONE", "HEADPHONES",
                        "MONITORS", "PANELS",
                    }),
                ],
            },
        ),
    ]
