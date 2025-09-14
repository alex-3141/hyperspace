dofile_once("data/scripts/lib/utilities.lua")

local id = GetUpdatedEntityID()

TAU = 6.28318530718

SpinMomentum = SpinMomentum or 0
TiltMomentum = TiltMomentum or 0
Spin = Spin or 0
Tilt = Tilt or 0
Yaw = Yaw or 0
YawMomentum = YawMomentum or 0
Transforming = Transforming or 0
Stretch = Stretch or 0
Morph = Morph or 0
Rotate = Rotate or 0

print("I ran")

math.fract = function(x)
	if(x >= 0) then
		return x - math.floor(x)
	else
		return x - math.ceil(x)
	end
end

math.sign = function(x)
	if(x < 0) then
		return -1
	else
		return 1
	end
end

function pushUniforms()
	local x, y = EntityGetTransform(id)
	GameSetPostFxParameter("hyperspace_matrix", x, y, Spin, Tilt)
	GameSetPostFxParameter("hyperspace_transition", Rotate, Yaw, Stretch, Morph)
end

function applyAngularMomentum(dx)
	Spin = Spin + SpinMomentum
	Tilt = Tilt + TiltMomentum
	Tilt = Tilt - Tilt * 0.2
	Tilt = Tilt * math.sign(dx or 1)
	Yaw = Yaw + YawMomentum
end

function buildAngularMomentum()
	local dx, dy = GameGetVelocityCompVelocity(id)

	SpinMomentum = SpinMomentum + dx / 5000
	SpinMomentum = SpinMomentum - SpinMomentum * 0.05
	TiltMomentum = TiltMomentum + dy / 5000
	TiltMomentum = TiltMomentum - TiltMomentum * 0.2
	applyAngularMomentum(dx)
end


function doHover()
	local x, y = EntityGetTransform(id)
	local float_range = 50
	local float_force = 4.5
	local float_sensor_sector = math.pi * 0.3

	local dir_x = 0
	local dir_y = float_range
	dir_x, dir_y = vec_rotate(dir_x, dir_y, ProceduralRandomf(x, y + GameGetFrameNum(), -float_sensor_sector, float_sensor_sector))

	local did_hit,hit_x,hit_y = RaytracePlatforms( x, y, x + dir_x, y + dir_y )
	if did_hit then
		local dist = get_distance(x, y, hit_x, hit_y)
		dist = math.max(6, dist) -- tame a bit on close encounters
		dir_x = -dir_x / dist * float_force
		dir_y = -dir_y / dist * float_force
		PhysicsApplyForce(id, dir_x, dir_y)
	end
end

function doSpinUp()
	Transforming = 1

	-- Cancel out off-axis spin
	if(math.abs(math.fract(Tilt / TAU)) < 0.01) then
		TiltMomentum = 0
		Tilt = 0
	else
		TiltMomentum = -math.fract(Tilt / TAU) * 3
	end

	if(math.abs(math.fract(Spin / TAU)) < 0.01) then
		SpinMomentum = 0
		Spin = 0
	else
		SpinMomentum = -math.fract(Spin / TAU) * 3
	end


	-- Positioning
	local target_x = 782
	local target_y = -1185
	local x, y = EntityGetTransform(id)
	local dx = target_x - x
	local dy = target_y - y

	EntitySetTransform( id, x + dx / 30, y + dy / 30 )

	if(math.abs(YawMomentum) < 0.5) then
		-- Build yaw momentum
		YawMomentum = YawMomentum + 0.001
	else
		GlobalsSetValue("HYPERSPACE_STATE", "MORPH")
	end

	applyAngularMomentum()
end

function doMorph()
	Morph = Morph + 0.005
	if(Morph > 1) then
		Morph = 1
		GlobalsSetValue("HYPERSPACE_STATE", "OPEN")
	end
	applyAngularMomentum()
end

function doOpen()
	Rotate = Rotate + 0.005
	if(Rotate > TAU) then
		Rotate = TAU
	end
end


if( GlobalsGetValue("HYPERSPACE_STATE") == "SPIN_UP" ) then
	doSpinUp()
	pushUniforms()
elseif ( GlobalsGetValue("HYPERSPACE_STATE") == "MORPH" ) then
	doMorph()
	pushUniforms()
elseif ( GlobalsGetValue("HYPERSPACE_STATE") == "OPEN" ) then
	doOpen()
	pushUniforms()
else
	buildAngularMomentum()
	doHover()
	pushUniforms()
end

