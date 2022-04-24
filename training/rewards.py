from utils.mybots_rewards import *
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward


def anneal_rewards_fn():
    max_steps = 10_000_000  # 20_000_000
    # when annealing, change the weights between 1 and 2, 2 is new
    reward1 = MyRewardFunction(
        team_spirit=0.1,
        goal_w=5,
        aerial_goal_w=10,
        double_tap_goal_w=0,
        shot_w=0.8,
        save_w=1.2,
        demo_w=0.2,
        above_w=0,
        got_demoed_w=-0.2,
        behind_ball_w=0.05,
        save_boost_w=0.003,
        concede_w=-5,
        velocity_w=0.1,
        velocity_pb_w=0.25,
        velocity_bg_w=1.0,
        aerial_ball_touch_w=15,
        kickoff_w=0.25,
        ball_touch_w=0.05,
    )

    reward2 = MyRewardFunction(
        team_spirit=0.2,
        goal_w=7,
        aerial_goal_w=10,
        double_tap_goal_w=0,
        shot_w=0.8,
        save_w=1.2,
        demo_w=1,
        above_w=0,
        got_demoed_w=-1,
        behind_ball_w=0.05,
        save_boost_w=0.05,
        concede_w=-7,
        velocity_w=0.2,
        velocity_pb_w=0.25,
        velocity_bg_w=1.5,
        aerial_ball_touch_w=15,
        kickoff_w=0.5,
        ball_touch_w=0,
    )

    alternating_rewards_steps = [reward1, max_steps, reward2]

    return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)


class MyRewardFunction(CombinedReward):
    def __init__(
            self,
            team_spirit=0.2,
            goal_w=10.0,
            aerial_goal_w=25.0,
            double_tap_goal_w=75.0,
            shot_w=0.2,
            save_w=5.0,
            demo_w=5.0,
            above_w=0.05,
            got_demoed_w=-6.0,
            behind_ball_w=0.01,
            save_boost_w=0.03,
            concede_w=-5.0,
            velocity_w=0.8,
            velocity_pb_w=0.5,
            velocity_bg_w=0.6,
            aerial_ball_touch_w=1.0,
            ball_touch_w=0.25,
            kickoff_w=0.5,
    ):
        self.team_spirit = team_spirit
        self.goal_w = goal_w
        self.aerial_goal_w = aerial_goal_w
        self.double_tap_goal_w = double_tap_goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.demo_w = demo_w
        self.above_w = above_w
        self.got_demoed_w = got_demoed_w
        self.behind_ball_w = behind_ball_w
        self.save_boost_w = save_boost_w
        self.concede_w = concede_w
        self.velocity_w = velocity_w
        self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w
        self.aerial_ball_touch_w = aerial_ball_touch_w
        self.ball_touch_w = ball_touch_w
        self.kickoff_w = kickoff_w
        # self.rewards = None
        goal_reward = EventReward(goal=self.goal_w, concede=self.concede_w)
        distrib_reward = DistributeRewards(goal_reward, team_spirit=self.team_spirit)
        super().__init__(
            reward_functions=(
                distrib_reward,
                AboveCrossbar(),
                SaveBoostReward(),
                VelocityReward(),
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                Demoed(),
                EventReward(
                    shot=self.shot_w,
                    save=self.save_w,
                    demo=self.demo_w,
                    touch=self.ball_touch_w,
                ),
                AerialRewardPerTouch(exp_base=1.75, max_touches_reward=50),
                AerialGoalReward(),
                DoubleTapReward(),
                KickoffReward(),
            ),
            reward_weights=(
                1.0,
                self.above_w,
                self.save_boost_w,
                self.velocity_w,
                self.velocity_pb_w,
                self.velocity_bg_w,
                self.got_demoed_w,
                1.0,
                self.aerial_ball_touch_w,
                self.aerial_goal_w,
                self.double_tap_goal_w,
                self.kickoff_w,
            )
        )


class MyOldRewardFunction(CombinedReward):
    def __init__(
            self,
            team_spirit=0.2,
            goal_w=10.0,
            aerial_goal_w=25.0,
            double_tap_goal_w=75.0,
            shot_w=0.2,
            save_w=5.0,
            demo_w=5.0,
            above_w=0.05,
            got_demoed_w=-6.0,
            behind_ball_w=0.01,
            save_boost_w=0.03,
            concede_w=-5.0,
            velocity_w=0.8,
            velocity_pb_w=0.5,
            velocity_bg_w=0.6,
            ball_touch_w=1.0,
            kickoff_w=0.5,
    ):
        self.team_spirit = team_spirit
        self.goal_w = goal_w
        self.aerial_goal_w = aerial_goal_w
        self.double_tap_goal_w = double_tap_goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.demo_w = demo_w
        self.above_w = above_w
        self.got_demoed_w = got_demoed_w
        self.behind_ball_w = behind_ball_w
        self.save_boost_w = save_boost_w
        self.concede_w = concede_w
        self.velocity_w = velocity_w
        self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w
        self.ball_touch_w = ball_touch_w
        self.kickoff_w = kickoff_w
        # self.rewards = None
        goal_reward = EventReward(goal=self.goal_w, concede=self.concede_w)
        distrib_reward = DistributeRewards(goal_reward, team_spirit=self.team_spirit)
        super().__init__(
            reward_functions=(
                distrib_reward,
                AboveCrossbar(),
                SaveBoostReward(),
                VelocityReward(),
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                Demoed(),
                EventReward(
                    shot=self.shot_w,
                    save=self.save_w,
                    demo=self.demo_w,
                ),
                AerialRewardPerTouch(exp_base=2, max_touches_reward=20),
                AerialGoalReward(),
                DoubleTapReward(),
                KickoffReward()
            ),
            reward_weights=(
                1.0,
                self.above_w,
                self.save_boost_w,
                self.velocity_w,
                self.velocity_pb_w,
                self.velocity_bg_w,
                self.got_demoed_w,
                1.0,
                self.ball_touch_w,
                self.aerial_goal_w,
                self.double_tap_goal_w,
                self.kickoff_w,
            )
        )

