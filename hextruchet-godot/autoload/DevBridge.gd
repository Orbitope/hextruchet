extends Node

## Dev-only loader for the MCP interaction bridge.
##
## The bridge (scripts/mcp_interaction_server.gd) is vendored tooling that opens
## a TCP port so an AI agent can drive the running game. It has no place in a
## shipped build, so the Web export filters the file out -- but an autoload
## pointing straight at a filtered-out script makes the export fail to start
## ("Failed to instantiate an autoload"), which silently breaks input.
##
## This thin wrapper is what gets autoloaded instead: it is always present, and
## it loads the bridge only when the file actually exists and we are not on web.
## Editor/desktop runs keep full MCP control; web builds quietly skip it.

const BRIDGE_PATH := "res://scripts/mcp_interaction_server.gd"


func _ready() -> void:
	if OS.has_feature("web"):
		return
	if not ResourceLoader.exists(BRIDGE_PATH):
		return
	var script: Resource = load(BRIDGE_PATH)
	if script == null:
		return
	var bridge: Node = (script as GDScript).new()
	bridge.name = "McpInteractionServer"
	add_child(bridge)
