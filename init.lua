dofile_once("mods/hyperspace/files/translations.lua").apply()
ModMaterialsFileAdd("mods/hyperspace/files/materials.xml")

Player = nil

function OnPlayerSpawned(player)
	Player = player
    local x, y = EntityGetTransform(player)
	EntityLoad("mods/hyperspace/files/entities/orthogonal_projection_matrix.xml", x, y)
end