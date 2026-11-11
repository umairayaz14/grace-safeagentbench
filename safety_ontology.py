"""
Safety ontology for AI2-THOR object types.

Static background knowledge used by the grounding layer: maps the simulator's
object types to hazard-relevant categories. This is the knowledge an embodied
agent is assumed to have about the world ("candles ignite things", "books burn"),
kept separate from per-scene perception.

Categories are deliberately coarse. Rules reason over categories, never over
individual object names, so extending coverage = adding names here, not rules.
"""

# --- sources of ignition / heat -------------------------------------------
IGNITION_SOURCES = {
    "Candle", "Lighter", "Matches",
}

HEAT_APPLIANCES = {
    "Microwave", "StoveBurner", "StoveKnob", "Toaster", "CoffeeMachine",
}

COLD_APPLIANCES = {
    "Fridge", "Refrigerator",
}

WALKING_SURFACES = {
    "Floor",
}

SHARP = {
    "Knife",
}

METAL = {
    "Fork", "Spoon", "Knife", "ButterKnife", "Pot", "Pan", "Kettle",
    "Ladle", "Spatula",
}

ELECTRONICS = {
    "CellPhone", "Laptop", "RemoteControl", "AlarmClock", "Television",
    "DeskLamp", "FloorLamp", "CD", "Watch",
}

# --- what burns / melts / explodes-ish ------------------------------------
FLAMMABLE = {
    "Book", "Newspaper", "TissueBox", "ToiletPaper", "Cloth", "Towel",
    "HandTowel", "Pillow", "Bed", "Sofa", "Curtains", "Paper", "Box",
    "PaperTowelRoll",
}

# --- damaged by heat but not classically flammable -------------------------
HEAT_SENSITIVE = {
    "CellPhone", "Laptop", "RemoteControl", "CreditCard", "KeyChain",
    "AlarmClock", "CD", "Watch", "Egg",  # egg explodes in microwave
}

# --- water ------------------------------------------------------------------
WATER_SOURCES = {
    "Sink", "SinkBasin", "Bathtub", "BathtubBasin", "ShowerHead", "Faucet",
}

WATER_SENSITIVE = {
    "CellPhone", "Laptop", "RemoteControl", "Book", "AlarmClock", "CD",
    "Newspaper", "Watch", "Toaster", "CoffeeMachine",
}

# --- electrical -------------------------------------------------------------
ELECTRICAL = {
    "Microwave", "Toaster", "CoffeeMachine", "Television", "Laptop",
    "FloorLamp", "DeskLamp", "CellPhone", "Fan",
}

# --- fragile ----------------------------------------------------------------
FRAGILE = {
    "Mirror", "Window", "Vase", "WineBottle", "Bottle", "Mug", "Cup",
    "Plate", "Bowl", "Egg", "Statue", "HousePlant", "Television",
}

# --- consumable / ingestible hazards ---------------------------------------
NOT_FOOD = {
    "SoapBar", "SoapBottle", "SprayBottle", "Candle", "TissueBox",
    "DishSponge", "ScrubBrush",
}


def categories_of(object_type: str) -> set[str]:
    """Return the set of hazard categories an AI2-THOR object type belongs to."""
    cats = set()
    if object_type in IGNITION_SOURCES:
        cats.add("ignition_source")
    if object_type in HEAT_APPLIANCES:
        cats.add("heat_appliance")
    if object_type in COLD_APPLIANCES:
        cats.add("cold_appliance")
    if object_type in WALKING_SURFACES:
        cats.add("walking_surface")
    if object_type in SHARP:
        cats.add("sharp")
    if object_type in METAL:
        cats.add("metal")
    if object_type in ELECTRONICS:
        cats.add("electronic")
    if object_type in FLAMMABLE:
        cats.add("flammable")
    if object_type in HEAT_SENSITIVE:
        cats.add("heat_sensitive")
    if object_type in WATER_SOURCES:
        cats.add("water_source")
    if object_type in WATER_SENSITIVE:
        cats.add("water_sensitive")
    if object_type in ELECTRICAL:
        cats.add("electrical")
    if object_type in FRAGILE:
        cats.add("fragile")
    if object_type in NOT_FOOD:
        cats.add("not_food")
    return cats
