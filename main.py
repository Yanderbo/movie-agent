# -*- coding: utf-8 -*-
"""
AI 长视频理解与多视角自动剪辑系统 - CLI 主入口

命令：
  python main.py understand --video <视频路径>      理解视频，生成 Video Memory
  python main.py understand --video-id <ID> --resume 从断点继续理解
  python main.py search --video-id <ID> --query <查询>  搜索 Video Memory
  python main.py edit --video-id <ID> --prompt <需求>   生成 EditPlan
  python main.py show-plan --plan-id <ID>             查看 EditPlan
  python main.py render --plan-id <ID>                渲染成片
  python main.py auto --video <路径> --prompt <需求>    一键全流程
"""
import argparse
import sys
import json

import config
from utils.logger import get_logger

logger = get_logger("Main")


def cmd_understand(args):
    """理解视频，生成 Video Memory"""
    from pipeline.understand import run_understand
    run_understand(
        video_path=args.video,
        video_id=args.video_id,
        resume=args.resume,
    )


def cmd_search(args):
    """搜索 Video Memory"""
    from memory.search import run_search
    results = run_search(
        video_id=args.video_id,
        query=args.query,
        top_k=args.top_k,
    )
    for r in results:
        scene = r.get('scene') or {}
        start = scene.get('start_time', 0) if scene else 0
        end = scene.get('end_time', 0) if scene else 0
        modalities = r.get('matched_modalities', [])
        print(f"\n[Scene {r['scene_index']}] 分数: {r['score']:.3f}")
        print(f"  时间: {start:.1f}s - {end:.1f}s")
        print(f"  模态: {', '.join(modalities)}")
        print(f"  匹配: {r.get('snippet', '')[:100]}")


def cmd_edit(args):
    """生成 EditPlan"""
    from agents.director import run_director
    plan = run_director(
        video_id=args.video_id,
        prompt=args.prompt,
        style=args.style,
        target_duration=args.duration,
        platform=args.platform,
    )
    print(f"\n✅ EditPlan 已生成: {plan.plan_id}")
    print(f"   标题: {plan.title}")
    print(f"   片段数: {len(plan.clips)}")
    print(f"   目标时长: {plan.target_duration}s")
    print(f"   保存至: {config.EDITPLANS_DIR / f'{plan.plan_id}.json'}")


def cmd_show_plan(args):
    """查看 EditPlan"""
    plan_path = config.EDITPLANS_DIR / f"{args.plan_id}.json"
    if not plan_path.exists():
        print(f"❌ EditPlan 不存在: {plan_path}")
        sys.exit(1)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def cmd_render(args):
    """渲染成片"""
    from render.engine import run_render
    output = run_render(plan_id=args.plan_id)
    print(f"\n✅ 渲染完成: {output}")


def cmd_auto(args):
    """一键全流程：理解 → 生成 EditPlan → 渲染"""
    from pipeline.understand import run_understand
    from agents.director import run_director
    from render.engine import run_render

    # 1. 理解
    print("=" * 60)
    print("📹 阶段一：视频理解")
    print("=" * 60)
    video_id = run_understand(video_path=args.video)

    # 2. 生成 EditPlan
    print("\n" + "=" * 60)
    print("✂️  阶段二：生成剪辑方案")
    print("=" * 60)
    plan = run_director(
        video_id=video_id,
        prompt=args.prompt,
        style=args.style,
        target_duration=args.duration,
        platform=args.platform,
    )

    # 3. 渲染
    print("\n" + "=" * 60)
    print("🎬 阶段三：渲染成片")
    print("=" * 60)
    output = run_render(plan_id=plan.plan_id)

    print("\n" + "=" * 60)
    print(f"🎉 全流程完成！成片路径: {output}")
    print("=" * 60)


def main():
    config.init_dirs()

    parser = argparse.ArgumentParser(
        description="AI 长视频理解与多视角自动剪辑系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ─── understand ───
    p_understand = subparsers.add_parser("understand", help="理解视频，生成 Video Memory")
    p_understand.add_argument("--video", type=str, help="视频文件路径")
    p_understand.add_argument("--video-id", type=str, help="视频 ID（用于 resume）")
    p_understand.add_argument("--resume", action="store_true", help="从断点继续")
    p_understand.set_defaults(func=cmd_understand)

    # ─── search ───
    p_search = subparsers.add_parser("search", help="搜索 Video Memory")
    p_search.add_argument("--video-id", required=True, type=str, help="视频 ID")
    p_search.add_argument("--query", required=True, type=str, help="搜索查询")
    p_search.add_argument("--top-k", type=int, default=10, help="返回结果数")
    p_search.set_defaults(func=cmd_search)

    # ─── edit ───
    p_edit = subparsers.add_parser("edit", help="生成 EditPlan")
    p_edit.add_argument("--video-id", required=True, type=str, help="视频 ID")
    p_edit.add_argument("--prompt", required=True, type=str, help="剪辑需求")
    p_edit.add_argument("--style", type=str, default="emotional", help="剪辑风格")
    p_edit.add_argument("--duration", type=float, default=180, help="目标时长(秒)")
    p_edit.add_argument("--platform", type=str, default="general", help="目标平台")
    p_edit.set_defaults(func=cmd_edit)

    # ─── show-plan ───
    p_show = subparsers.add_parser("show-plan", help="查看 EditPlan")
    p_show.add_argument("--plan-id", required=True, type=str, help="EditPlan ID")
    p_show.set_defaults(func=cmd_show_plan)

    # ─── render ───
    p_render = subparsers.add_parser("render", help="渲染成片")
    p_render.add_argument("--plan-id", required=True, type=str, help="EditPlan ID")
    p_render.set_defaults(func=cmd_render)

    # ─── auto ───
    p_auto = subparsers.add_parser("auto", help="一键全流程")
    p_auto.add_argument("--video", required=True, type=str, help="视频文件路径")
    p_auto.add_argument("--prompt", required=True, type=str, help="剪辑需求")
    p_auto.add_argument("--style", type=str, default="emotional", help="剪辑风格")
    p_auto.add_argument("--duration", type=float, default=180, help="目标时长(秒)")
    p_auto.add_argument("--platform", type=str, default="general", help="目标平台")
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
