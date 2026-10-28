"""Execute pinned upstream goal code on Boolean unit traces, not benchmark data."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.evaluation.libero_mem_metrics import LiberoMemEvaluator
from src.evaluation.libero_mem_protocol import load_protocol, task_spec

ROOT=Path(__file__).resolve().parents[2]


def upstream_fixture(target=3):
    root=ROOT/'external/libero-mem'
    protocol=load_protocol(source_root=root)
    source=root/'libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_check_success','_eval_goal')]
    scope={}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    e=SimpleNamespace(_satisfied_subgoals=[],_sub_goal_live_time=0,_sub_goal_nonlive_time=0,_overshot=False,predicate=True)
    e.parsed_problem={'goal_state':[['sequence']+[['and',['on','bowl','plate']] for _ in range(target)]]}
    e._eval_predicate=lambda _:e.predicate
    for name,fn in scope.items():
        if name in ('_check_success','_eval_goal'):setattr(e,name,fn.__get__(e))
    return e


def tick(e,predicate):
    e.predicate=predicate
    e._check_success()  # upstream reward call
    e._check_success()  # upstream step call
    return e._check_success(inc=True)


def test_intermediate_true_is_not_full_sequence_or_terminal_success():
    e=upstream_fixture();metric=LiberoMemEvaluator('T3',3)
    for i in range(7):tick(e,True)
    for i in range(16):tick(e,False)
    e.predicate=True
    assert e._check_success()  # the reward read can commit the intermediate milestone
    e._check_success()
    e._check_success(inc=True)
    r=metric.observe(24,e,True)
    assert r.completed_subgoals==1 and r.scr==pytest.approx(1/3) and not r.full_success


def test_terminal_requires_complete_final_predicate_and_no_overshoot():
    e=upstream_fixture(1);metric=LiberoMemEvaluator('T1',1)
    for _ in range(7):tick(e,True)
    for _ in range(16):tick(e,False)
    tick(e,True)
    assert metric.observe(24,e,True).full_success is False
    assert metric.observe(25,e,True,terminal=True).full_success
    e._overshot=True
    assert not metric.observe(26,e,True,terminal=True).full_success


def test_upstream_counter_is_cumulative_and_inc_false_can_commit():
    e=upstream_fixture(1)
    for _ in range(6):e._check_success(inc=True)
    e.predicate=False
    for _ in range(16):e._check_success(inc=True)
    # Source does not reset live counter on False; inc=False may append a goal.
    assert e._sub_goal_live_time==6
    e.predicate=True
    assert e._check_success(inc=False)
    assert len(e._satisfied_subgoals)==1
    for _ in range(7):tick(e,False)
    assert e._overshot
    assert not tick(e,True)


def test_metric_observation_is_read_only_and_ordered():
    e=upstream_fixture();metric=LiberoMemEvaluator('T3',3)
    before=(list(e._satisfied_subgoals),e._sub_goal_live_time,e._sub_goal_nonlive_time,e._overshot)
    metric.observe(0,e,True)
    assert before==(e._satisfied_subgoals,e._sub_goal_live_time,e._sub_goal_nonlive_time,e._overshot)
    with pytest.raises(ValueError):metric.observe(0,e,True)


def test_protocol_is_explicitly_custom_and_excludes_unsupported_tasks():
    p=load_protocol()
    assert not p['official_paper_comparable']
    assert task_spec(p,'T1')['target_cycles']==1
    assert task_spec(p,'T3')['target_cycles']==3
    with pytest.raises(ValueError):task_spec(p,'T10')
