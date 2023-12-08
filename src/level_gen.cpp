#include "level_gen.hpp"
#include <iostream>

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

enum class RoomType : uint32_t {
    SingleButton,
    DoubleButton,
    CubeBlocking,
    CubeButtons,
    Key,
    NumTypes,
};

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.rand() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.rand() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
}

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    const int32_t numRooms = ctx.singleton<RoomCount>().count;
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    if ((ctx.data().simFlags & SimFlags::UseComplexLevel)
        != SimFlags::UseComplexLevel)
    {
        // Create the outer wall entities
        // Behind
        ctx.data().borders[0] = ctx.makeEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[0],
            Vector3{
                0,
                -consts::wallWidth / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::worldWidth + consts::wallWidth * 2,
                consts::wallWidth,
                2.f,
            });

        // Right
        ctx.data().borders[1] = ctx.makeEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[1],
            Vector3{
                consts::worldWidth / 2.f + consts::wallWidth / 2.f,
                consts::roomLength * numRooms / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::roomLength * numRooms,
                2.f,
            });

        // Left
        ctx.data().borders[2] = ctx.makeEntity<PhysicsEntity>();
        setupRigidBodyEntity(
            ctx,
            ctx.data().borders[2],
            Vector3{
                -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
                consts::roomLength * numRooms / 2.f,
                0,
            },
            Quat{1, 0, 0, 0},
            SimObject::Wall,
            EntityType::Wall,
            ResponseType::Static,
            Diag3x3{
                consts::wallWidth,
                consts::roomLength * numRooms,
                2.f,
            });
    }

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] = ctx.makeEntity<Agent>();

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<GrabState>(agent).constraintEntity = Entity::none();
        ctx.get<EntityType>(agent) = EntityType::Agent;
        ctx.get<AgentID>(agent).id = (int32_t)i;
    }

    // Populate OtherAgents component, which maintains a reference to the
    // other agents in the world for each agent.
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];

        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < consts::numAgents; j++) {
            if (i == j) {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

    if ((ctx.data().simFlags & SimFlags::UseComplexLevel)
        != SimFlags::UseComplexLevel)
    {
        for (CountT i = 0; i < 3; i++) {
            Entity wall_entity = ctx.data().borders[i];
            registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
        }
    }

    const int32_t numRooms = ctx.singleton<RoomCount>().count;
     auto second_rng = RNG::make(ctx.data().curEpisodeIdx);
     // VISHNU MOD: RETRIEVE PROGRESS SO FAR
     //printf("About to access illegally\n");
     GlobalProgress& progress = ctx.singleton<GlobalProgress>();
     //printf("About to access illegally %f \n", *(progress.progressPtr));
     // VISHNU MOD: what if we place them in any room we want
     float max_room = 0.999 + (int)(fmax(0.f, fmin(float(numRooms), (*(progress.progressPtr)) / consts::roomLength))); 
     int rand_room = ((int)(second_rng.rand() * max_room));
     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         // VISHNU MOD: LOG PROGRESS SO FAR
         *(progress.progressPtr) = fmax(*(progress.progressPtr), ctx.get<Position>(agent_entity)[1]);

         ctx.get<viz::VizCamera>(agent_entity) =
             viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                 1.5f * math::up, (int32_t)i);

         // Place the agents near the starting wall
         Vector3 pos {
             randInRangeCentered(ctx, 
                 consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
             randBetween(ctx, consts::agentRadius * 1.1f,  2.f),
             0.f,
         };

        if ((ctx.data().simFlags & SimFlags::StartInDiscoveredRooms) ==
                SimFlags::StartInDiscoveredRooms) {
            pos[1] += consts::roomLength * rand_room;
            if (rand_room > 0) {
               float range = consts::worldWidth / 2.f - 2.5f * consts::agentRadius;
               float x_pos = second_rng.rand() * range - range / 2.f;
               pos[0] = x_pos;
            }
        }

         if (i % 2 == 0) {
             pos.x += consts::worldWidth / 4.f;
         } else {
             pos.x -= consts::worldWidth / 4.f;
         }

         ctx.get<Position>(agent_entity) = pos;
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             randInRangeCentered(ctx, math::pi / 4.f),
             math::up);

         auto &grab_state = ctx.get<GrabState>(agent_entity);
         if (grab_state.constraintEntity != Entity::none()) {
             ctx.destroyEntity(grab_state.constraintEntity);
             grab_state.constraintEntity = Entity::none();
         }

         ctx.get<Progress>(agent_entity).maxY = pos.y;

         ctx.get<Velocity>(agent_entity) = {
             Vector3::zero(),
             Vector3::zero(),
         };
         ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
         ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
         ctx.get<Action>(agent_entity) = Action {
             .moveAmount = 0,
             .moveAngle = 0,
             .rotate = consts::numTurnBuckets / 2,
             .interact = 0,
         };

         ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
         ctx.get<KeyCode>(agent_entity).code = 0;
     }
}

static Entity makeKey(Engine &ctx,
                         float key_x,
                         float key_y,
                         int32_t code)
{
    Entity key = ctx.makeEntity<KeyEntity>();
    ctx.get<Position>(key) = Vector3 {
        key_x,
        key_y,
        std::sqrt(3.0f) * consts::keyWidth,
    };
    ctx.get<Rotation>(key) = Quat::angleAxis(math::pi / 4.0f, Vector3{0,1,0}) * Quat::angleAxis(math::pi / 4.0f, Vector3{1,0,0});
    ctx.get<Scale>(key) = Diag3x3 {
        consts::keyWidth,
        consts::keyWidth,
        consts::keyWidth,
    };
    ctx.get<ObjectID>(key) = ObjectID { (int32_t)SimObject::Key };
    ctx.get<KeyState>(key).claimed = false;
    ctx.get<KeyState>(key).code.code = code;
    ctx.get<EntityType>(key) = EntityType::Key;

    return key;
}

static void setupDoor(Engine &ctx,
                      Entity door,
                      Span<const Entity> buttons,
                      KeyCode key_code,
                      bool is_persistent)
{
    DoorProperties &props = ctx.get<DoorProperties>(door);

    for (CountT i = 0; i < buttons.size(); i++) {
        props.buttons[i] = buttons[i];
    }
    props.numButtons = (int32_t)buttons.size();
    props.isPersistent = is_persistent;

    ctx.get<KeyCode>(door).code = key_code.code;
}

// Builds the two walls & door that block the end of the challenge room
static void makeWall(Engine &ctx,
                    Room &room,
                    CountT room_idx,
                    Vector3 pos,
                    int32_t orientation)
{
    // No door wall
    // Entity wall = ctx.makeEntity<PhysicsEntity>();
    // setupRigidBodyEntity(
    //     ctx,
    //     wall,
    //     pos,
    //     Quat { 1, 0, 0, 0 },
    //     SimObject::Wall,
    //     EntityType::Wall,
    //     ResponseType::Static,
    //     Diag3x3 {
    //         consts::roomLength,
    //         consts::wallWidth,
    //         1.75f,
    // });

    // // Wall with a door.

    // ctx.get<Rotation>(wall) = Quat::angleAxis((orientation) * math::pi * 0.5, math::up);

    // registerRigidBodyEntity(ctx, wall, SimObject::Wall);

    // room.walls[orientation] = wall;

    // return;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        pos + Vector3 {
            orientation % 2 == 0 ? (-consts::roomLength + left_len) / 2.f : 0,
            orientation % 2 != 0 ? (-consts::roomLength + left_len) / 2.f : 0,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    ctx.get<Rotation>(left_wall) = Quat::angleAxis((orientation) * math::pi * 0.5, math::up);
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        pos + Vector3 {
            orientation % 2 == 0 ? (consts::roomLength - right_len) / 2.f : 0,
            orientation % 2 != 0 ? (consts::roomLength - right_len) / 2.f : 0,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    ctx.get<Rotation>(right_wall) = Quat::angleAxis((orientation) * math::pi * 0.5, math::up);
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);

    Entity door = ctx.makeEntity<DoorEntity>();
    setupRigidBodyEntity(
        ctx,
        door,
        pos + Vector3 {
            orientation % 2 == 0 ? door_center - consts::worldWidth / 2.f : 0,
            orientation % 2 != 0 ? door_center - consts::worldWidth / 2.f : 0,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        EntityType::Door,
        ResponseType::Static,
        Diag3x3 {
            consts::doorWidth * 0.8f,
            consts::wallWidth,
            1.75f,
        });
    ctx.get<Rotation>(door) = Quat::angleAxis((orientation) * math::pi * 0.5, math::up);
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;
    ctx.get<KeyCode>(door).code = 1 << orientation; // Default requires no keys.

        float key_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::keyWidth);
    float key_y = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::keyWidth);
    Entity key = makeKey(ctx, key_x, key_y, 1 << orientation); // key code
    setupDoor(ctx, door, {}, ctx.get<KeyState>(key).code, true);

    room.walls[2 * orientation] = left_wall;
    room.walls[2 * orientation + 1] = right_wall;
    room.door[orientation] = door;
    room.entities[orientation] = key;
}


// Builds the two walls & door that block the end of the challenge room
static void makeEndWall(Engine &ctx,
                        Room &room,
                        CountT room_idx)
{
    float y_pos = consts::roomLength * (room_idx + 1) -
        consts::wallWidth / 2.f;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            (-consts::worldWidth + left_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            (consts::worldWidth - right_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);

    Entity door = ctx.makeEntity<DoorEntity>();
    setupRigidBodyEntity(
        ctx,
        door,
        Vector3 {
            door_center - consts::worldWidth / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        EntityType::Door,
        ResponseType::Static,
        Diag3x3 {
            consts::doorWidth * 0.8f,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;
    ctx.get<KeyCode>(door).code = 0; // Default requires no keys.

    room.walls[0] = left_wall;
    room.walls[1] = right_wall;
    room.door[0] = door;
}

static Entity makeButton(Engine &ctx,
                         float button_x,
                         float button_y)
{
    Entity button = ctx.makeEntity<ButtonEntity>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        consts::buttonWidth,
        consts::buttonWidth,
        0.2f,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObject::Button };
    ctx.get<ButtonState>(button).isPressed = false;
    ctx.get<EntityType>(button) = EntityType::Button;

    return button;
}

static Entity makeCube(Engine &ctx,
                       float cube_x,
                       float cube_y,
                       float scale = 1.f)
{
    Entity cube = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        cube,
        Vector3 {
            cube_x,
            cube_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Cube,
        EntityType::Cube,
        ResponseType::Dynamic,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, cube, SimObject::Cube);

    return cube;
}

// A room with a door unlocked by a key.
// Door opens to an agent with the key.
static CountT makeKeyRoom(Engine &ctx,
                          Room &room,
                          Room &placementRoom,
                          float y_min,
                          float y_max,
                          int32_t code)
{
    float key_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::keyWidth);
    float key_y = randBetween(ctx, y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::keyWidth / 2.f);

    Entity key = makeKey(ctx, key_x, key_y, code); // key code

    setupDoor(ctx, room.door[0], {}, ctx.get<KeyState>(key).code, true);

    CountT openEntityIdx = 0;
    while (openEntityIdx < consts::maxEntitiesPerRoom) {
        if (placementRoom.entities[openEntityIdx] == Entity::none()) {
            placementRoom.entities[openEntityIdx] = key;
            break;
        }
        ++openEntityIdx;
    }

    return (openEntityIdx == 0);
}

// A room with a single button that needs to be pressed, the door stays open.
static CountT makeSingleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::buttonWidth);
    float button_y = randBetween(ctx, y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity button = makeButton(ctx, button_x, button_y);

    setupDoor(ctx, room.door[0], { button }, KeyCode{ 0 }, true);

    room.entities[0] = button;

    return 1;
}

// A room with two buttons that need to be pressed simultaneously,
// the door stays open.
static CountT makeDoubleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth);

    float a_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity a = makeButton(ctx, a_x, a_y);

    float b_x = randBetween(ctx,
        consts::buttonWidth,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float b_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity b = makeButton(ctx, b_x, b_y);

    setupDoor(ctx, room.door[0], { a, b }, KeyCode{ 0 }, true);

    room.entities[0] = a;
    room.entities[1] = b;

    return 2;
}

// This room has 3 cubes blocking the exit door as well as two buttons.
// The agents either need to pull the middle cube out of the way and
// open the door or open the door with the buttons and push the cubes
// into the next room.
static CountT makeCubeBlockingRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door[0], { button_a, button_b }, KeyCode{ 0 }, true);

    Vector3 door_pos = ctx.get<Position>(room.door[0]);

    float cube_a_x = door_pos.x - 3.f;
    float cube_a_y = door_pos.y - 2.f;

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = door_pos.x;
    float cube_b_y = door_pos.y - 2.f;

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    float cube_c_x = door_pos.x + 3.f;
    float cube_c_y = door_pos.y - 2.f;

    Entity cube_c = makeCube(ctx, cube_c_x, cube_c_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;
    room.entities[4] = cube_c;

    return 5;
}

// This room has 2 buttons and 2 cubes. The buttons need to remain pressed
// for the door to stay open. To progress, the agents must push at least one
// cube onto one of the buttons, or more optimally, both.
static CountT makeCubeButtonsRoom(Engine &ctx,
                                  Room &room,
                                  float y_min,
                                  float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door[0], { button_a, button_b }, KeyCode{ 0 }, false);

    float cube_a_x = randBetween(ctx,
        -consts::worldWidth / 4.f,
        -1.5f);

    float cube_a_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = randBetween(ctx,
        1.5f,
        consts::worldWidth / 4.f);

    float cube_b_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;

    return 4;
}

// Make the doors and separator walls at the end of the room
// before delegating to specific code based on room_type.
static void makeComplexRoom(Engine &ctx,
                     LevelState &level,
                     CountT room_idx,
                     RoomType room_type,
                     const RoomRep *roomList,
                     Entity *doorList)
{

    Room &room = level.rooms[room_idx];
    // Need to set any extra entities to type none so random uninitialized data
    // from prior episodes isn't exported to pytorch as agent observations.
    for (CountT i = 0; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i] = Entity::none();
    }

    for (CountT i = 0; i < 8; i++) {
        room.walls[i] = Entity::none();
    }

    for (CountT i = 0; i < 4; i++) {
        room.door[i] = Entity::none();
    }

    RoomRep this_room = roomList[room_idx];

    float room_center_x = consts::roomLength * this_room.x;
    float room_center_y = consts::roomLength * this_room.y;

    // N, E, S, W
    int32_t wallsToGenerate[4] = {1, 1, 1, 1};
    for (int i = 0; i < consts::maxRooms; ++i) {
        if (roomList[i].y == this_room.y + 1) {
            wallsToGenerate[0] = 0;
        }
        if (roomList[i].x == this_room.x + 1) {
            wallsToGenerate[1] = 0;
        }
        if (roomList[i].y == this_room.y - 1) {
            wallsToGenerate[2] = 0;
        }
        if (roomList[i].x == this_room.x - 1) {
            wallsToGenerate[3] = 0;
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (wallsToGenerate[i] == 0) {
            continue;
        }

        Vector3 wallCenter;
        switch (i) {
            case 0:
                wallCenter = Vector3{room_center_x, room_center_y + consts::roomLength * 0.5f, 0.0f};
                break;
            case 1:
                wallCenter = Vector3{room_center_x+ consts::roomLength * 0.5f, room_center_y, 0.0f};
                break;
            case 2:
                wallCenter = Vector3{room_center_x, room_center_y - consts::roomLength * 0.5f, 0.0f};
                break;
            case 3:
                wallCenter = Vector3{room_center_x - consts::roomLength * 0.5f, room_center_y, 0.0f};
                break;
            default: break;
        }
        makeWall(ctx, room, room_idx, wallCenter, i);
    }

    return;
    // 1. Room centerpoint from door position and orientation. 
    //    Figure out where on the grid that puts you.
    // 2. Get a picture of the surrounding rooms from the grid. 
    //    Use that to get constraints on which walls need to be built.
    // 3. Randomly decide which walls get doors, and build those.
    // 4. Keep going until you've placed the total number of rooms.
    // 5. Resolve key doors by scattering keys to open rooms. For each new key, update the open rooms list.



    makeEndWall(ctx, room, room_idx);

    float room_y_min = room_idx * consts::roomLength;
    float room_y_max = (room_idx + 1) * consts::roomLength;

    switch (room_type) {
    case RoomType::SingleButton: {
        makeSingleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::DoubleButton: {
        makeDoubleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeBlocking: {
        makeCubeBlockingRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeButtons: {
        makeCubeButtonsRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::Key: {
        CountT randomRoomIdx = (int32_t)randBetween(ctx, 0.0f, float(room_idx));
        // Put the key in a random already generate room.
        room_y_min = randomRoomIdx * consts::roomLength;
        room_y_max = (randomRoomIdx + 1) * consts::roomLength;
        makeKeyRoom(ctx, room, level.rooms[randomRoomIdx], room_y_min, room_y_max, 1 << room_idx);
    } break;
    default: MADRONA_UNREACHABLE();
    }


}


// Make the doors and separator walls at the end of the room
// before delegating to specific code based on room_type.
static void makeRoom(Engine &ctx,
                     LevelState &level,
                     CountT room_idx,
                     RoomType room_type)
{
    
    Room &room = level.rooms[room_idx];
    // Need to set any extra entities to type none so random uninitialized data
    // from prior episodes isn't exported to pytorch as agent observations.
    for (CountT i = 0; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i] = Entity::none();
    }


    makeEndWall(ctx, room, room_idx);

    float room_y_min = room_idx * consts::roomLength;
    float room_y_max = (room_idx + 1) * consts::roomLength;

    switch (room_type) {
    case RoomType::SingleButton: {
        makeSingleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::DoubleButton: {
        makeDoubleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeBlocking: {
        makeCubeBlockingRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeButtons: {
        makeCubeButtonsRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::Key: {
        CountT randomRoomIdx = (int32_t)randBetween(ctx, 0.0f, float(room_idx));
        // Put the key in a random already generate room.
        room_y_min = randomRoomIdx * consts::roomLength;
        room_y_max = (randomRoomIdx + 1) * consts::roomLength;
        makeKeyRoom(ctx, room, level.rooms[randomRoomIdx], room_y_min, room_y_max, 1 << room_idx);
    } break;
    default: MADRONA_UNREACHABLE();
    }


}

static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    // For training simplicity, define a fixed sequence of levels.
    makeRoom(ctx, level, 0, RoomType::DoubleButton);
    makeRoom(ctx, level, 1, RoomType::CubeBlocking);
    makeRoom(ctx, level, 2, RoomType::CubeButtons);

#if 0
    // An alternative implementation could randomly select the type for each
    // room rather than a fixed progression of challenge difficulty
    for (CountT i = 0; i < consts::numRooms; i++) {
        RoomType room_type = (RoomType)(
            ctx.data().rng.rand() * (uint32_t)RoomType::NumTypes);

        makeRoom(ctx, level, i, room_type);
    }
#endif
}

static void generateComplexLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    Entity doorList[consts::maxRooms * 4];
    int32_t doorIdx = 0;
    for (int i = 0; i < consts::maxRooms * 4; ++i) {
        doorList[i] = Entity::none();
    }

    RoomRep roomList[consts::maxRooms];
    for (int i = 0; i < consts::maxRooms; ++i) {
        roomList[i].x = 2 * consts::maxRooms;
        roomList[i].y = 2 * consts::maxRooms;
        roomList[i].door = Entity::none();
    }

    // First room. Agents start in this room.
    roomList[0].x = 0;
    roomList[0].y = 0;

    // RoomType is interpreted as DoorType.
    makeComplexRoom(ctx, level, 0, RoomType::DoubleButton, roomList, &doorList[doorIdx]);

    //makeRoom(ctx, level, 0, RoomType::DoubleButton);
    //makeRoom(ctx, level, 1, RoomType::CubeBlocking);
    //makeRoom(ctx, level, 2, RoomType::CubeButtons);
    //makeRoom(ctx, level, 3, RoomType::Key);

    // // An alternative implementation could randomly select the type for each
    // // room rather than a fixed progression of challenge difficulty
    // for (CountT i = 0; i < consts::numRooms; i++) {
    //     RoomType room_type = (RoomType)(
    //         ctx.data().rng.rand() * (uint32_t)RoomType::NumTypes);

    //     makeRoom(ctx, level, i, room_type);
    // }
}


// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);

    if ((ctx.data().simFlags & SimFlags::UseComplexLevel) ==
        SimFlags::UseComplexLevel)
    {
        generateComplexLevel(ctx);
    }
    else
    {
        generateLevel(ctx);
    }

    printf("Generated World!\n");
}

}
