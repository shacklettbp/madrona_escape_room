#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;



#define CKPT_ENABLED

namespace madEscape {

// Helper function for determining room membership.
static inline CountT roomIndex(const Position &p)
{
    return std::max(CountT(0),
    std::min(consts::numRooms - 1, CountT(p.y / consts::roomLength)));
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

static inline float angleObs(float v)
{
    return v / math::pi;
}

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<GrabState>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<RoomEntityObservations>();
    registry.registerComponent<DoorObservation>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<LevelState>();
    registry.registerSingleton<CheckpointState>();
    registry.registerSingleton<CheckpointIndices>();

    // TODO: restore debugging
    registry.registerSingleton<DummySingleton>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();

    // TODO: ZM, guessing we will eventually we want this to make checkpoint
    // state visible to the training code.
    registry.exportSingleton<DummySingleton>(
        (uint32_t)ExportID::Dummy);
    registry.exportSingleton<CheckpointIndices>(
        (uint32_t)ExportID::Checkpoint);
    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Agent, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);
    registry.exportColumn<Agent, RoomEntityObservations>(
        (uint32_t)ExportID::RoomEntityObservations);
    registry.exportColumn<Agent, DoorObservation>(
        (uint32_t)ExportID::DoorObservation);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
}

static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities
    LevelState &level = ctx.singleton<LevelState>();
    for (CountT i = 0; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        for (CountT j = 0; j < consts::maxEntitiesPerRoom; j++) {
            if (room.entities[j] != Entity::none()) {
                ctx.destroyEntity(room.entities[j]);
            }
        }

        ctx.destroyEntity(room.walls[0]);
        ctx.destroyEntity(room.walls[1]);
        ctx.destroyEntity(room.door);
    }
}

static inline void initWorld(Engine &ctx)
{
    if (ctx.data().enableVizRender) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);
    ctx.data().curEpisodeIdx = episode_idx;
    
    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

inline void loadCheckpointSystem(Engine &ctx, CheckpointState &ckptState) 
{
    //printf("Should restore %lu\n", (unsigned long)&ctx.data());
    printf("load checkpoint %d\n", ctx.singleton<CheckpointIndices>().loadCheckpoint);
    int ckptIdx = ctx.singleton<CheckpointIndices>().currentCheckpointIdx;
    int doCkpt = ctx.singleton<CheckpointIndices>().loadCheckpoint;
    if (doCkpt == 0 || ckptIdx == 0) {
        // No reset, or first reset so nothing to load from checkpoint.
        // reset will be zeroed by the checkpoint system.
        return;
    }


    // Default value restores to the previous state.
    // Training code can override this.
    int chosenIdx = ckptIdx - 1;

    printf("Restored ckpt %d\n", chosenIdx);
    // Load the checkpoint from the beginning of the episode.
    Checkpoint &ckpt = ckptState.checkpoints[chosenIdx]; // Get the previous state.
    {
        // Agent parameters: physics state, grabstate
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptAgentQuery, 
            [&](Position &p, Rotation &r, Velocity &v, Reward &re, Done &d, StepsRemaining &s, Progress &pr)
            {
                p  = ckpt.agentStates[idx].p;
                r  = ckpt.agentStates[idx].r;
                v  = ckpt.agentStates[idx].v;
                re = ckpt.agentStates[idx].re;
                d  = ckpt.agentStates[idx].d;
                s  = ckpt.agentStates[idx].s;
                pr = ckpt.agentStates[idx].pr;
                idx++;
                printf("Restored agent %d position = %f, %f, %f\n", idx, p.x, p.y, p.z);
            }
        );
    }

    {
        // Door parameters
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptDoorQuery, 
            [&](Position &p, Rotation &r, Velocity &v, OpenState& o)
            {
                p = ckpt.doorStates[idx].p;
                r = ckpt.doorStates[idx].r;
                v = ckpt.doorStates[idx].v;
                o = ckpt.doorStates[idx].o;
                idx++;
            }
        );
    }

    {
        // Correct the walls now that we have the doors
        ctx.iterateQuery(ctx.data().ckptWallQuery,
            [&](Position &p, Scale &s, EntityType &e)
            {
                if (e == EntityType::Wall)
                {
                    for (int i = 0; i < consts::numRooms; ++i)
                    {
                        // Check all doors.
                        Position door_pos = ckpt.doorStates[i].p;
                        constexpr float doorWidth = consts::worldWidth / 3.f;

                        if (door_pos.y == p.y)
                        {
                            float door_center = door_pos.x + consts::worldWidth / 2.f;
                            // We found one of two wall pairs for this door
                            if (p.x < 0.f)
                            {
                                // Left door
                                float left_len = door_center - 0.5f * doorWidth;
                                p.x = (-consts::worldWidth + left_len) / 2.f;
                                s = Diag3x3{
                                    left_len,
                                    consts::wallWidth,
                                    1.75f,
                                };
                            }
                            else
                            {
                                // Right door
                                float right_len = consts::worldWidth - door_center - 0.5f * doorWidth;
                                p.x = (consts::worldWidth - right_len) / 2.f;
                                s = Diag3x3{
                                    right_len,
                                    consts::wallWidth,
                                    1.75f,
                                };
                            }
                        }
                    }
                }
            });
    }

    
    {
        // Cubes
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptCubeQuery, 
            [&](Position &p, Rotation &r, Velocity &v, EntityType &e)
            {
                if (e == EntityType::Cube) {
                    p = ckpt.cubeStates[idx].p;
                    r = ckpt.cubeStates[idx].r;
                    v = ckpt.cubeStates[idx].v;
                    idx++;
                }
            }
        );
    }

    {
        // Buttons
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptButtonQuery, 
            [&](Position &p, Rotation &r, ButtonState &b)
            {
                p = ckpt.buttonStates[idx].p;
                r = ckpt.buttonStates[idx].r;
                b = ckpt.buttonStates[idx].b;
                idx++;
            }
        );
    }
}

inline void checkpointSystem(Engine &ctx, CheckpointState &ckptState)
{
    // This runs every frame
    int32_t loaded_checkpoint = ctx.singleton<CheckpointIndices>().loadCheckpoint;
    if (loaded_checkpoint == 1) {
        // The world was just reset, we should not checkpoint
        // Check if we need to reset the checkpoint index.
        // We can't do this during the resetSystem because we need the
        // checkpoint index during loading.
        ctx.iterateQuery(ctx.data().ckptAgentQuery, 
        [&](Position &p, Rotation &r, Velocity &v, Reward &re, Done &d, StepsRemaining &s, Progress &pr)
        {
            // Reset the checkpoint index
            ctx.singleton<CheckpointIndices>().currentCheckpointIdx = (consts::maxCheckpoints - s.t) % consts::maxCheckpoints;
        });
        printf("Reset checkpoint idx to %d\n", ctx.singleton<CheckpointIndices>().currentCheckpointIdx);
        ctx.singleton<CheckpointIndices>().loadCheckpoint = 0;
        return;
    }

    int ckptIdx = ctx.singleton<CheckpointIndices>().currentCheckpointIdx++;

    bool update = 
     ckptIdx % 40 == 0 ||
    (ckptIdx + 1) % 40 == 0 ||
    (ckptIdx - 1) % 40 == 0;

    if (true || update) {
     //   printf("writing ckpt idx %d\n", ckptIdx);
    }


    // Grab current checkpoint slot (per world).
    Checkpoint &ckpt = ckptState.checkpoints[ckptIdx];
    {
        // Agent parameters: physics state, reward, done.
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptAgentQuery, 
            [&](Position &p, Rotation &r, Velocity &v, Reward &re, Done &d, StepsRemaining &s, Progress &pr)
            {
                ckpt.agentStates[idx].p = p;
                ckpt.agentStates[idx].r = r;
                ckpt.agentStates[idx].v = v;
                ckpt.agentStates[idx].re = re;
                ckpt.agentStates[idx].d = d;
                ckpt.agentStates[idx].s = s;
                ckpt.agentStates[idx].pr = pr;
                idx++;
                if (update) {
                  // printf("Agent %d position = %f, %f, %f\n", idx, p.x, p.y, p.z);
                }
            }
        );
    }

    {
        // Door parameters
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptDoorQuery,
            [&](Position &p, Rotation &r, Velocity &v, OpenState& o)
            {
                ckpt.doorStates[idx].p = p;
                ckpt.doorStates[idx].r = r;
                ckpt.doorStates[idx].v = v;
                ckpt.doorStates[idx].o = o;
                idx++;
            }
        );
    }

    {
        // Cubes
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptCubeQuery, 
            [&](Position &p, Rotation &r, Velocity &v, EntityType &e)
            {
                if (e == EntityType::Cube) {
                    ckpt.cubeStates[idx].p = p;
                    ckpt.cubeStates[idx].r = r;
                    ckpt.cubeStates[idx].v = v;
                    idx++;
                }
            }
        );
    }

    {
        // Buttons
        int idx = 0;
        ctx.iterateQuery(ctx.data().ckptButtonQuery, 
            [&](Position &p, Rotation &r, ButtonState &b)
            {
                ckpt.buttonStates[idx].p = p;
                ckpt.buttonStates[idx].r = r;
                ckpt.buttonStates[idx].b = b;
                idx++;
            }
        );
    }
}

inline void debugSystem(Engine &ctx, DummySingleton &ds)
{
    printf("dummy %d\n", ds.dummy);
    //ctx.singleton<DummySingleton>().dummy = 7;
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{

    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 9; // TODO: restore
        cleanupWorld(ctx);
        initWorld(ctx);

        if (ctx.data().enableVizRender) {
            viz::VizRenderingSystem::markEpisode(ctx);
        }
    }
}

// Translates discrete actions from the Action component to forces
// used by the physics simulation.
inline void movementSystem(Engine &,
                           Action &action, 
                           Rotation &rot, 
                           ExternalForce &external_force,
                           ExternalTorque &external_torque)
{
    constexpr float move_max = 1000;
    constexpr float turn_max = 320;

    Quat cur_rot = rot;

    float move_amount = action.moveAmount *
        (move_max / (consts::numMoveAmountBuckets - 1));

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(action.moveAngle) * move_angle_per_bucket;

    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    constexpr float turn_delta_per_bucket = 
        turn_max / (consts::numTurnBuckets / 2);
    float t_z =
        turn_delta_per_bucket * (action.rotate - consts::numTurnBuckets / 2);

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}

// Implements the grab action by casting a short ray in front of the agent
// and creating a joint constraint if a grabbable entity is hit.
inline void grabSystem(Engine &ctx,
                       Entity e,
                       Position pos,
                       Rotation rot,
                       Action action,
                       GrabState &grab)
{
    if (action.grab == 0) {
        return;
    }

    // if a grab is currently in progress, triggering the grab action
    // just releases the object
    if (grab.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab.constraintEntity);
        grab.constraintEntity = Entity::none();
        
        return;
    } 

    // Get the per-world BVH singleton component
    auto &bvh = ctx.singleton<broadphase::BVH>();
    float hit_t;
    Vector3 hit_normal;

    Vector3 ray_o = pos + 0.5f * math::up;
    Vector3 ray_d = rot.rotateVec(math::fwd);

    Entity grab_entity =
        bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.0f);

    if (grab_entity == Entity::none()) {
        return;
    }

    auto response_type = ctx.get<ResponseType>(grab_entity);
    if (response_type != ResponseType::Dynamic) {
        return;
    }

    auto entity_type = ctx.get<EntityType>(grab_entity);
    if (entity_type == EntityType::Agent) {
        return;
    }

    Entity constraint_entity = ctx.makeEntity<ConstraintData>();
    grab.constraintEntity = constraint_entity;

    Vector3 other_pos = ctx.get<Position>(grab_entity);
    Quat other_rot = ctx.get<Rotation>(grab_entity);

    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

    Vector3 hit_pos = ray_o + ray_d * hit_t;
    Vector3 r2 =
        other_rot.inv().rotateVec(hit_pos - other_pos);

    Quat attach1 = { 1, 0, 0, 0 };
    Quat attach2 = (other_rot.inv() * rot).normalize();

    float separation = hit_t - 1.25f;

    ctx.get<JointConstraint>(constraint_entity) = JointConstraint::setupFixed(
        e, grab_entity, attach1, attach2, r1, r2, separation);
}

// Animates the doors opening and closing based on OpenState
inline void setDoorPositionSystem(Engine &,
                                  Position &pos,
                                  OpenState &open_state)
{
    if (open_state.isOpen) {
        // Put underground
        if (pos.z > -4.5f) {
            pos.z += -consts::doorSpeed * consts::deltaT;
        }
    }
    else if (pos.z < 0.0f) {
        // Put back on surface
        pos.z += consts::doorSpeed * consts::deltaT;
    }
    
    if (pos.z >= 0.0f) {
        pos.z = 0.0f;
    }
}


// Checks if there is an entity standing on the button and updates
// ButtonState if so.
inline void buttonSystem(Engine &ctx,
                         Position pos,
                         ButtonState &state)
{
    AABB button_aabb {
        .pMin = pos + Vector3 { 
            -consts::buttonWidth / 2.f, 
            -consts::buttonWidth / 2.f,
            0.f,
        },
        .pMax = pos + Vector3 { 
            consts::buttonWidth / 2.f, 
            consts::buttonWidth / 2.f,
            0.25f
        },
    };

    bool button_pressed = false;
    RigidBodyPhysicsSystem::findEntitiesWithinAABB(
            ctx, button_aabb, [&](Entity) {
        button_pressed = true;
    });

    state.isPressed = button_pressed;
}

// Check if all the buttons linked to the door are pressed and open if so.
// Optionally, close the door if the buttons aren't pressed.
inline void doorOpenSystem(Engine &ctx,
                           OpenState &open_state,
                           const DoorProperties &props)
{
    bool all_pressed = true;
    for (int32_t i = 0; i < props.numButtons; i++) {
        Entity button = props.buttons[i];
        all_pressed = all_pressed && ctx.get<ButtonState>(button).isPressed;
    }

    if (all_pressed) {
        open_state.isOpen = true;
    } else if (!props.isPersistent) {
        open_state.isOpen = false;
    }
}

// Make the agents easier to control by zeroing out their velocity
// after each step.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

static inline float distObs(float v)
{
    return v / consts::worldLength;
}

static inline float globalPosObs(float v)
{
    return v / consts::worldLength;
}

// Translate xy delta to polar observations for learning.
static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

static inline float encodeType(EntityType type)
{
    return (float)type / (float)EntityType::NumTypes;
}

// This system packages all the egocentric observations together 
// for the policy inputs.
inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      Rotation rot,
                                      const Progress &progress,
                                      const GrabState &grab,
                                      SelfObservation &self_obs,
                                      PartnerObservations &partner_obs,
                                      RoomEntityObservations &room_ent_obs,
                                      DoorObservation &door_obs)
{
    const CountT cur_room_idx = roomIndex(pos);

    self_obs.roomX = pos.x / (consts::worldWidth / 2.f);
    self_obs.roomY = (pos.y - cur_room_idx * consts::roomLength) /
        consts::roomLength;
    self_obs.globalX = globalPosObs(pos.x);
    self_obs.globalY = globalPosObs(pos.y);
    self_obs.globalZ = globalPosObs(pos.z);
    self_obs.maxY = globalPosObs(progress.maxY);
    self_obs.theta = angleObs(computeZAngle(rot));
    self_obs.isGrabbing = grab.constraintEntity != Entity::none() ?
        1.f : 0.f;

    assert(!isnan(self_obs.roomX));
    assert(!isnan(self_obs.roomY));
    assert(!isnan(self_obs.globalX));
    assert(!isnan(self_obs.globalY));
    assert(!isnan(self_obs.globalZ));
    assert(!isnan(self_obs.maxY));
    assert(!isnan(self_obs.theta));
    assert(!isnan(self_obs.isGrabbing));

    //int ckptIdx = ctx.singleton<CheckpointIndices>().currentCheckpointIdx;

    //printf("SelfObs roomx %f\n", self_obs.roomX);
    //printf("SelfObs roomY %f\n", self_obs.roomY);
    //printf("SelfObs globalX %f\n", self_obs.globalX);
    //printf("SelfObs globalY %f\n", self_obs.globalY);
    //printf("SelfObs globalZ %f\n", self_obs.globalZ);
    //printf("SelfObs maxY %f\n", self_obs.maxY);
    //printf("SelfObs theta %f\n", self_obs.theta);
    //printf("SelfObs isGrabbing %f\n", self_obs.isGrabbing);


    Quat to_view = rot.inv();

    // Context::iterateQuery() runs serially over archteypes
    // matching the components on which it is templated.
    {
        int idx = 0; // Context::iterateQuery() is serial, so this is safe.
        ctx.iterateQuery(ctx.data().otherAgentQuery,
            [&](Position &other_pos, GrabState &other_grab) {
                Vector3 to_other = other_pos - pos;

                // Detect whether or not the other agent is the current agent.
                if (to_other.length() == 0.0f) {
                    return;
                }

                partner_obs.obs[idx++] = {
                    .polar = xyToPolar(to_view.rotateVec(to_other)),
                    .isGrabbing = other_grab.constraintEntity != Entity::none() ? 1.f : 0.f,
                };
            });
    }

    // Becase we iterate by component matching, we can encounter entities
    // that are in the current room, have EntityType::None, but were never 
    // Cubes or Buttons, so shouldn't be allowed to zero entries of the 
    // RoomEntityObservations table. Therefore, we zero the table manually 
    // here, and let only the current Cubes and Buttons that exist update it.
    EntityObservation ob;
    ob.polar = { 0.f, 1.f };
    ob.encodedType = encodeType(EntityType::None);
    for (int i = 0; i < consts::maxObservationsPerAgent; ++i) {
       room_ent_obs.obs[i] = ob;
    }
    
    {
       int idx = 0;
        ctx.iterateQuery(ctx.data().roomEntityQuery, [&](Position &p, EntityType &e) {
            // We want information on cubes and buttons in the current room.
            if (roomIndex(p) != cur_room_idx ||
                (e != EntityType::Cube && e != EntityType::Button)) {
                return;
            }

            EntityObservation ob;
            Vector3 to_entity = p - pos;
            ob.polar = xyToPolar(to_view.rotateVec(to_entity));
            ob.encodedType = encodeType(e);

            if (idx < consts::maxObservationsPerAgent) {
                room_ent_obs.obs[idx++] = ob;
            }
        });
    }

    // Context.query() version.
    ctx.iterateQuery(ctx.data().doorQuery, [&](Position &p, OpenState &os) {
       if (roomIndex(p) != cur_room_idx) {
           return;
       }
       door_obs.polar = xyToPolar(to_view.rotateVec(p - pos));
       door_obs.isOpen = os.isOpen ? 1.f : 0.f;
    });
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (
            float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
            };
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar.samples[idx] = {
                .depth = distObs(hit_t),
                .encodedType = encodeType(entity_type),
            };
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
}

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    // Just in case agents do something crazy, clamp total reward
    float reward_pos = fminf(pos.y, consts::worldLength * 2);

    float old_max_y = progress.maxY;

    float new_progress = reward_pos - old_max_y;

    float reward;
    if (new_progress > 0) {
        reward = new_progress * consts::rewardPerDist;
        progress.maxY = reward_pos;
    } else {
        reward = consts::slackReward;
    }

    out_reward.v = reward;
}

// Each agent gets a small bonus to it's reward if the other agent has
// progressed a similar distance, to encourage them to cooperate.
// This system reads the values of the Progress component written by
// rewardSystem for other agents, so it must run after.
inline void bonusRewardSystem(Engine &ctx,
                              OtherAgents &others,
                              Progress &progress,
                              Reward &reward)
{
    bool partners_close = true;
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = others.e[i];
        Progress other_progress = ctx.get<Progress>(other);

        if (fabsf(other_progress.maxY - progress.maxY) > 2.f) {
            partners_close = false;
        }
    }

    if (partners_close && reward.v > 0.f) {
        reward.v *= 1.25f;
    }
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1) {
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }

}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

// Build the task graph
void Sim::setupTasks(TaskGraphBuilder &builder, const Config &cfg)
{

    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            Rotation,
            ExternalForce,
            ExternalTorque
        >>({});

    // Scripted door behavior
    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            OpenState
        >>({move_sys});

    // Build BVH for broadphase / raycasting
    auto broadphase_setup_sys =
        phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder, 
                                                           {set_door_pos_sys});

    // Grab action, post BVH build to allow raycasting
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
        grabSystem,
            Entity,
            Position,
            Rotation,
            Action,
            GrabState
        >>({broadphase_setup_sys});

    // Physics collision detection and solver
    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {grab_sys}, consts::numPhysicsSubsteps);

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // Check buttons
    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            ButtonState
        >>({phys_done});

    // Set door to start opening if button conditions are met
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
            OpenState,
            DoorProperties
        >>({button_sys});

    // Compute initial reward now that physics has updated the world state
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward
        >>({door_open_sys});

    // Assign partner's reward
    auto bonus_reward_sys = builder.addToGraph<ParallelForNode<Engine,
         bonusRewardSystem,
            OtherAgents,
            Progress,
            Reward
        >>({reward_sys});


    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({bonus_reward_sys});

    // Conditionally reset the world if the episode is over
    auto debug_sys = builder.addToGraph<ParallelForNode<Engine,
        debugSystem,
        DummySingleton
        >>({done_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
        WorldReset
        >>({debug_sys});

#ifdef CKPT_ENABLED
    // Conditionally load the checkpoint here including Done, Reward, 
    // and StepsRemaining. With Observations this should reconstruct 
    // all state that the training code needs.
    // This runs after the reset system resets the world.
    auto load_checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        loadCheckpointSystem,
        CheckpointState
        >>({reset_sys});

    // Conditionally checkpoint the state of the system if we are on the Nth step.
    auto checkpoint_sys = builder.addToGraph<ParallelForNode<Engine,
        checkpointSystem,
        CheckpointState
        >>({load_checkpoint_sys});
#endif

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({
#ifdef CKPT_ENABLED
        checkpoint_sys
#else
        reset_sys
#endif
    });
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({
#ifdef CKPT_ENABLED
        checkpoint_sys
#else
        reset_sys
#endif
    });
    (void)recycle_sys;
#endif

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {
#ifdef CKPT_ENABLED
        checkpoint_sys
#else
        reset_sys
#endif
        });

    // The lidar system
#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
        >>({post_reset_broadphase});

    if (cfg.enableViewer) {
        viz::VizRenderingSystem::setupTasks(builder, {reset_sys});
    }

#ifdef MADRONA_GPU_MODE
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {lidar});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    auto sort_buttons = queueSortByWorld<ButtonEntity>(
        builder, {sort_phys_objects});
    auto sort_walls = queueSortByWorld<DoorEntity>(
        builder, {sort_buttons});
    auto sort_constraints = queueSortByWorld<ConstraintData>(
        builder, {sort_walls});
    (void)sort_walls;
#else
    (void)lidar;
#endif

    // Finally, collect observations for the next step.
    builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            Progress,
            GrabState,
            SelfObservation,
            PartnerObservations,
            RoomEntityObservations,
            DoorObservation
        >>({post_reset_broadphase, 
#ifdef MADRONA_GPU_MODE
        sort_constraints
#endif
        });
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +
        4; // side walls + floor

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities, max_total_entities * max_total_entities / 2,
        consts::numAgents);

    enableVizRender = cfg.enableViewer;

    if (enableVizRender) {
        viz::VizRenderingSystem::init(ctx, init.vizBridge);
    }

    autoReset = cfg.autoReset;

    // Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // Generate initial world state
    initWorld(ctx);

    // Create the queries for collectObservations
    ctx.data().otherAgentQuery = ctx.query<Position, GrabState>();
    ctx.data().roomEntityQuery = ctx.query<Position, EntityType>();
    ctx.data().doorQuery       = ctx.query<Position, OpenState>();

    // Create the queries for checkpointing.
    ctx.data().ckptAgentQuery = ctx.query<Position, Rotation, Velocity, Reward, Done, StepsRemaining, Progress>();
    ctx.data().ckptDoorQuery = ctx.query<Position, Rotation, Velocity, OpenState>();
    ctx.data().ckptCubeQuery = ctx.query<Position, Rotation, Velocity, EntityType>();
    ctx.data().ckptButtonQuery = ctx.query<Position, Rotation, ButtonState>();
    ctx.data().ckptWallQuery = ctx.query<Position, Scale, EntityType>();

    // Initialize checkpointing state
    //CheckpointState &ckptState = ctx.singleton<CheckpointState>();
    ctx.singleton<CheckpointIndices>().currentCheckpointIdx = 5; // TODO: restore
    ctx.singleton<CheckpointIndices>().loadCheckpoint = 0;
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
