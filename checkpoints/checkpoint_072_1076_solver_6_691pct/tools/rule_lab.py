from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arc_solver.types import Grid, Rule
from src.arc_solver.grid_utils import as_grid, as_rows, is_valid_grid, dedupe_grids
from src.arc_solver.predict import predict_task
from src.arc_solver.scoring import score_task_attempts, attempt_matches
from tools.rule_generators import FAMILY_ALIASES, FAMILY_FITTERS



@dataclass
class ScoreSummary:
    solved_outputs: int = 0
    total_outputs: int = 0
    task_normalized: float = 0.0
    fully_solved_tasks: int = 0
    partially_solved_tasks: int = 0

    def add_task(self, hits: int, total: int) -> None:
        self.solved_outputs += hits
        self.total_outputs += total
        if total:
            self.task_normalized += hits / total
            if hits == total:
                self.fully_solved_tasks += 1
            elif hits > 0:
                self.partially_solved_tasks += 1

    @property
    def output_percent(self) -> float:
        return 100.0 * self.solved_outputs / self.total_outputs if self.total_outputs else 0.0


def print_score_summary(label: str, summary: ScoreSummary) -> None:
    print(f"{label}:")
    print(f"  raw solved outputs: {summary.solved_outputs}/{summary.total_outputs}")
    print(f"  output score: {summary.output_percent:.3f}%")
    print(f"  task-normalized score: {summary.task_normalized:.3f}")
    print(f"  fully solved tasks: {summary.fully_solved_tasks}")
    print(f"  partially solved tasks: {summary.partially_solved_tasks}")
    print(f"  solved individual outputs: {summary.solved_outputs}")


def validate_arc_grid_strict(grid) -> tuple[bool, str | None]:
    try:
        if not isinstance(grid, (list, tuple)):
            return False, "invalid_rank"
        if len(grid) == 0:
            return False, "empty_grid"
        if len(grid) > 30:
            return False, "invalid_shape"
        width = None
        for row in grid:
            if not isinstance(row, (list, tuple)):
                return False, "invalid_rank"
            if len(row) == 0:
                return False, "invalid_shape"
            if width is None:
                width = len(row)
                if width > 30:
                    return False, "invalid_shape"
            elif len(row) != width:
                return False, "ragged_rows"
            for cell in row:
                if isinstance(cell, bool) or not isinstance(cell, int):
                    return False, "invalid_dtype"
                if cell < 0 or cell > 9:
                    return False, "invalid_color"
        return True, None
    except Exception:
        return False, "stage_exception"


def _strict_grid_or_none(grid) -> Grid | None:
    ok, _ = validate_arc_grid_strict(grid)
    if not ok:
        return None
    return tuple(tuple(row) for row in grid)

@dataclass
class CandidateStats:
    family: str
    name: str
    hits: int = 0
    new: int = 0
    overlap: int = 0
    tasks: set[str] = field(default_factory=set)
    errors: int = 0
    invalid_reasons: Counter[str] = field(default_factory=Counter)
    loo: dict[str, str] = field(default_factory=dict)
    support: dict[str, tuple[int, list[str]]] = field(default_factory=dict)
    residuals: dict[str, str] = field(default_factory=dict)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_families(names: list[str] | None) -> list[str]:
    if not names:
        return list(FAMILY_FITTERS)
    selected: list[str] = []
    for name in names:
        key = FAMILY_ALIASES.get(name, name)
        if key not in FAMILY_FITTERS:
            valid = ", ".join(sorted(set(FAMILY_FITTERS) | set(FAMILY_ALIASES)))
            raise SystemExit(f"Unknown family '{name}'. Valid families/aliases: {valid}")
        if key not in selected:
            selected.append(key)
    return selected


def train_pairs(task: dict) -> list[tuple[Grid, Grid]]:
    return [(as_grid(p["input"]), as_grid(p["output"])) for p in task["train"]]


def candidate_attempts(rule: Rule, task: dict) -> tuple[list[dict], int, Counter[str]]:
    attempts: list[dict] = []
    errors = 0
    invalid_reasons: Counter[str] = Counter()
    for item in task["test"]:
        preds: list[Grid] = []
        try:
            pred = rule.predict(as_grid(item["input"]))
        except Exception:
            pred = None
            ok, reason = False, "stage_exception"
        else:
            ok, reason = validate_arc_grid_strict(pred)
        if ok:
            preds.append(_strict_grid_or_none(pred))  # type: ignore[arg-type]
        else:
            errors += 1
            invalid_reasons[reason or "stage_exception"] += 1
        preds = dedupe_grids([g for g in preds if g is not None])
        while len(preds) < 2:
            preds.append(((0,),))
        attempts.append({"attempt_1": as_rows(preds[0]), "attempt_2": as_rows(preds[1])})
    return attempts, errors, invalid_reasons


def hit_indices(attempts: list[dict], expected_outputs: list) -> set[int]:
    return {i for i, _ in enumerate(expected_outputs) if i < len(attempts) and attempt_matches(attempts[i], expected_outputs[i])}


def _grid_sig(g: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in g)


def _loo_status(family: str, rule_name: str, train: list[tuple[Grid, Grid]]) -> str:
    if len(train) < 2:
        return "partial"
    passed = 0
    total = 0
    for i, (held_inp, held_out) in enumerate(train):
        subset = train[:i] + train[i + 1:]
        try:
            rules = FAMILY_FITTERS[family](subset)
        except Exception:
            total += 1
            continue
        matches = [r for r in rules if r.name == rule_name]
        total += 1
        ok = False
        for rule in matches:
            try:
                ok = rule.predict(held_inp) == held_out
            except Exception:
                ok = False
            if ok:
                break
        if ok:
            passed += 1
    if passed == total and total:
        return "yes"
    if passed == 0:
        return "no"
    return "partial"


def _color_map_repairable(pred: Grid, exp: Grid) -> bool:
    if not is_valid_grid(pred) or not is_valid_grid(exp) or len(pred) != len(exp) or len(pred[0]) != len(exp[0]):
        return False
    mapping: dict[int, int] = {}
    for r, row in enumerate(pred):
        for c, v in enumerate(row):
            dst = exp[r][c]
            if v in mapping and mapping[v] != dst:
                return False
            mapping[v] = dst
    return True


def _residual_summary(pred: Grid | None, exp_rows) -> str:
    exp = as_grid(exp_rows)
    if not is_valid_grid(pred):
        return "shape_match=no mismatch_count=? mismatch_bbox=- added_cells_count=? deleted_cells_count=? recolored_cells_count=? foreground_mask_agreement=no color_map_repairable=no"
    p = pred  # type: ignore[assignment]
    shape_match = len(p) == len(exp) and (not p or not exp or len(p[0]) == len(exp[0]))
    if not shape_match:
        return "shape_match=no mismatch_count=? mismatch_bbox=- added_cells_count=? deleted_cells_count=? recolored_cells_count=? foreground_mask_agreement=no color_map_repairable=no"
    mism=[]; added=deleted=recolored=fg_agree=fg_total=0
    for r,row in enumerate(p):
        for c,v in enumerate(row):
            e=exp[r][c]
            if (v != 0) == (e != 0): fg_agree += 1
            fg_total += 1
            if v != e:
                mism.append((r,c))
                if v == 0 and e != 0: added += 1
                elif v != 0 and e == 0: deleted += 1
                else: recolored += 1
    bbox = "-" if not mism else f"({min(r for r,_ in mism)},{min(c for _,c in mism)})-({max(r for r,_ in mism)},{max(c for _,c in mism)})"
    fg = f"{fg_agree}/{fg_total}"
    repair = "yes" if _color_map_repairable(p, exp) else "no"
    return f"shape_match=yes mismatch_count={len(mism)} mismatch_bbox={bbox} added_cells_count={added} deleted_cells_count={deleted} recolored_cells_count={recolored} foreground_mask_agreement={fg} color_map_repairable={repair}"



def _shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def _palette(g: Grid) -> set[int]:
    return {v for row in g for v in row}


def _components(g: Grid, bg: int = 0) -> list[set[tuple[int, int]]]:
    h, w = _shape(g); seen=set(); comps=[]
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg or (r,c) in seen: continue
            color=g[r][c]; stack=[(r,c)]; seen.add((r,c)); comp=set()
            while stack:
                rr,cc=stack.pop(); comp.add((rr,cc))
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc=rr+dr,cc+dc
                    if 0<=nr<h and 0<=nc<w and (nr,nc) not in seen and g[nr][nc]==color:
                        seen.add((nr,nc)); stack.append((nr,nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int,int]]) -> tuple[int,int,int,int] | None:
    if not cells: return None
    return min(r for r,_ in cells), min(c for _,c in cells), max(r for r,_ in cells), max(c for _,c in cells)


def _is_rect(cells: set[tuple[int,int]]) -> bool:
    b=_bbox(cells)
    return bool(b) and len(cells)==(b[2]-b[0]+1)*(b[3]-b[1]+1)


def delta_diagnostics(inp: Grid, out: Grid) -> list[str]:
    if _shape(inp) != _shape(out): return []
    h,w=_shape(inp); changed={(r,c) for r in range(h) for c in range(w) if inp[r][c]!=out[r][c]}
    if not changed: return ["no_change"]
    feats=[]
    if _is_rect(changed): feats.append("one_rectangle")
    comps=_mask_components(changed)
    if len(comps)==1: feats.append("one_connected_component")
    if all(len({r for r,_ in comp})==1 for comp in comps): feats.append("straight_horizontal_segments")
    if all(len({c for _,c in comp})==1 for comp in comps): feats.append("straight_vertical_segments")
    pfeats = _periodic_position_features(changed, inp, out)
    feats.extend([f for f in ("exact_nontrivial_period", "row_arithmetic_progression", "column_arithmetic_progression", "lattice_completion", "translated_copy_pattern") if f in pfeats])
    if not any(f in pfeats for f in ("exact_nontrivial_period", "row_arithmetic_progression", "column_arithmetic_progression", "lattice_completion", "translated_copy_pattern")) and "trivial_period_only" in pfeats:
        feats.append("trivial_period_only")
    in_comps=[frozenset(c) for c in _components(inp)]
    if any(changed==set(c) for c in in_comps): feats.append("component_masks")
    if any(_bbox(changed)==_bbox(set(c)) for c in in_comps): feats.append("component_bounding_boxes")
    singletons=[next(iter(c)) for c in _components(inp) if len(c)==1]
    if singletons and ("straight_horizontal_segments" in feats or "straight_vertical_segments" in feats): feats.append("rays_from_singleton_like_cells")
    return feats


def _mask_components(cells: set[tuple[int,int]]) -> list[set[tuple[int,int]]]:
    unseen=set(cells); comps=[]
    while unseen:
        start=unseen.pop(); stack=[start]; comp={start}
        while stack:
            r,c=stack.pop()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                p=(r+dr,c+dc)
                if p in unseen:
                    unseen.remove(p); comp.add(p); stack.append(p)
        comps.append(comp)
    return comps



def _periodic_position_features(cells: set[tuple[int,int]], inp: Grid, out: Grid) -> set[str]:
    h,w=_shape(inp); feats=set()
    if len(cells)<3:
        return feats
    rows=sorted({r for r,_ in cells}); cols=sorted({c for _,c in cells})
    def ap(xs):
        if len(xs)<3: return False
        ds=[b-a for a,b in zip(xs,xs[1:])]
        return len(set(ds))==1 and ds[0]>0
    if ap(rows): feats.add("row_arithmetic_progression")
    if ap(cols): feats.add("column_arithmetic_progression")
    for pr in range(1,h):
        if h//pr < 2: continue
        ok=True
        for r,c in cells:
            rr=r+pr
            if rr<h and (rr,c) not in cells: ok=False; break
        if ok and any(r+pr<h for r,c in cells): feats.add("exact_nontrivial_period")
    for pc in range(1,w):
        if w//pc < 2: continue
        ok=True
        for r,c in cells:
            cc=c+pc
            if cc<w and (r,cc) not in cells: ok=False; break
        if ok and any(c+pc<w for r,c in cells): feats.add("exact_nontrivial_period")
    if len(rows)*len(cols)==len(cells) and len(rows)>1 and len(cols)>1:
        feats.add("lattice_completion")
    comps=_mask_components(cells)
    sigs=[]
    for comp in comps:
        b=_bbox(comp)
        if b:
            r0,c0,_,_=b; sigs.append(frozenset((r-r0,c-c0,out[r][c]) for r,c in comp))
    if len(sigs)!=len(set(sigs)) and len(sigs)>1:
        feats.add("translated_copy_pattern")
    if not feats:
        feats.add("trivial_period_only")
    return feats

def _comp_infos(g: Grid):
    infos=[]
    for comp in _components(g):
        colors=Counter(g[r][c] for r,c in comp)
        infos.append({"cells":comp,"bbox":_bbox(comp),"size":len(comp),"colors":dict(colors)})
    return infos

def _grid_text(g: Grid)->str:
    return "\n".join("".join(str(v) for v in row) for row in g)

def _local_strip(g: Grid, rows: bool, cols: bool) -> Grid | None:
    h,w=_shape(g)
    sr={r for r,row in enumerate(g) if row and row[0]!=0 and all(v==row[0] for v in row)} if rows else set()
    sc={c for c in range(w) if h and g[0][c]!=0 and all(g[r][c]==g[0][c] for r in range(h))} if cols else set()
    out=[[v for c,v in enumerate(row) if c not in sc] for r,row in enumerate(g) if r not in sr]
    return tuple(tuple(row) for row in out) if out and out[0] else None

def _extract_candidate_kinds(inp: Grid, out: Grid):
    kinds=[]; ih,iw=_shape(inp); oh,ow=_shape(out)
    if oh<=ih and ow<=iw:
        for r0 in range(ih-oh+1):
            for c0 in range(iw-ow+1):
                sub=tuple(tuple(inp[r][c] for c in range(c0,c0+ow)) for r in range(r0,r0+oh))
                if sub==out: kinds.append(f"crop@({r0},{c0})")
    for info in _comp_infos(inp):
        b=info["bbox"]
        if b:
            r0,c0,r1,c1=b
            sub=tuple(tuple(inp[r][c] for c in range(c0,c1+1)) for r in range(r0,r1+1))
            if sub==out: kinds.append(f"component_bbox@{b}")
    if _local_strip(inp, True, True)==out: kinds.append("separator_lines_removed")
    if _local_strip(inp, True, False)==out: kinds.append("separator_rows_removed")
    if _local_strip(inp, False, True)==out: kinds.append("separator_cols_removed")
    return kinds or ["no"]

def inspect_residual_task(task_id: str, challenges: dict, show_grids: bool=False) -> None:
    if task_id not in challenges: raise SystemExit(f"Task not found: {task_id}")
    task=challenges[task_id]
    print(f"Task {task_id}: train_pairs={len(task.get('train', []))}")
    for i,pair in enumerate(task.get('train', [])):
        inp=as_grid(pair['input']); out=as_grid(pair['output']); h,w=_shape(inp); oh,ow=_shape(out)
        print(f"\nTrain {i}: input shape={h}x{w} output shape={oh}x{ow}")
        print(f"input palette={sorted(_palette(inp))} output palette={sorted(_palette(out))}")
        for label,g in (("input",inp),("output",out)):
            infos=_comp_infos(g); print(f"{label} 4-connected components={len(infos)}")
            for j,info in enumerate(infos[:30]): print(f"  {label} comp {j}: bbox={info['bbox']} size={info['size']} colors={info['colors']}")
        if (h,w)==(oh,ow):
            changed={(r,c) for r in range(h) for c in range(w) if inp[r][c]!=out[r][c]}
            added={(r,c) for r,c in changed if inp[r][c]==0 and out[r][c]!=0}; deleted={(r,c) for r,c in changed if inp[r][c]!=0 and out[r][c]==0}; recol={(r,c) for r,c in changed if inp[r][c] and out[r][c] and inp[r][c]!=out[r][c]}
            print(f"changed count={len(changed)} added count={len(added)} deleted count={len(deleted)} recolored count={len(recol)} changed bbox={_bbox(changed)}")
            print(f"added connected components={[{'bbox':_bbox(c),'size':len(c),'colors':dict(Counter(out[r][cc] for r,cc in c))} for c in _mask_components(added)]}")
            in_comps=_comp_infos(inp); preserved=all(all(out[r][c]==inp[r][c] for r,c in info['cells']) for info in in_comps)
            print(f"existing components preserved exactly={preserved}")
            touches=[]
            for idx,info in enumerate(in_comps):
                n=sum(1 for r,c in added for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)) if (r+dr,c+dc) in info['cells'])
                if n: touches.append((idx,n))
            print(f"added cells touch existing component={bool(touches)} details={touches}")
            print(f"added cells bridge two existing components={len({i for i,n in touches if n})>=2}")
            print(f"added cells extend rows/columns from existing cells={bool(added and (set(r for r,_ in added)&{r for info in in_comps for r,_ in info['cells']} or set(c for _,c in added)&{c for info in in_comps for _,c in info['cells']}))}")
            if added and not deleted and not recol:
                print(f"added cells by color={dict(Counter(out[r][c] for r,c in added))} added-component bbox={[_bbox(c) for c in _mask_components(added)]}")
                print(f"added rows={sorted({r for r,_ in added})} added columns={sorted({c for _,c in added})}")
                dists=[]
                for a,b in combinations(in_comps,2): dists.append(min(abs(r-r2)+abs(c-c2) for r,c in a['cells'] for r2,c2 in b['cells']))
                md=min(dists) if dists else None; print(f"minimum Manhattan distance between input components={md}")
                print(f"additions form a shortest Manhattan bridge={md is not None and len(added)==max(0,md-1) and len(touches)>=2}")
                print(f"additions form a full bbox edge={_is_rect(added) and (len({r for r,_ in added})==1 or len({c for _,c in added})==1)}")
                print(f"additions complete a rectangle={_is_rect({(r,c) for info in in_comps for r,c in info['cells']}|added)}")
                masks=[frozenset((r-min(x for x,_ in info['cells']),c-min(y for _,y in info['cells'])) for r,c in info['cells']) for info in in_comps]
                am=[frozenset((r-min(x for x,_ in c),cc-min(y for _,y in c)) for r,cc in c) for c in _mask_components(added)]
                print(f"additions repeat an existing component mask={bool(set(masks)&set(am))}")
        print(f"output equals crop/panel/component/separator extraction={_extract_candidate_kinds(inp,out)}")
        if show_grids:
            print('input grid:\n'+_grid_text(inp)); print('output grid:\n'+_grid_text(out))
def _periodic_positions(cells: set[tuple[int,int]], h: int, w: int) -> bool:
    if len(cells) < 3: return False
    for pr in range(1, min(6,h)+1):
        for pc in range(1, min(6,w)+1):
            buckets={(r%pr,c%pc) for r,c in cells}
            if 0 < len(buckets) <= max(1, len(cells)//3): return True
    return False


def _exact_small_period(g: Grid) -> bool:
    h,w=_shape(g)
    for pr in range(1, min(6,h)+1):
        for pc in range(1, min(6,w)+1):
            if (pr,pc)==(h,w): continue
            if all(g[r][c]==g[r%pr][c%pc] for r in range(h) for c in range(w)):
                return True
    return False


def _separator_features(g: Grid) -> tuple[bool,bool,bool,list[tuple[int,int]]]:
    h,w=_shape(g)
    rows=[r for r,row in enumerate(g) if row[0]!=0 and all(v==row[0] for v in row)] if h and w else []
    cols=[c for c in range(w) if g[0][c]!=0 and all(g[r][c]==g[0][c] for r in range(h))] if h and w else []
    panels=[]
    if len(cols)==1:
        spans=[(0,cols[0]),(cols[0]+1,w)]; panels=[(h,b-a) for a,b in spans if a<b]
    if len(rows)==1:
        spans=[(0,rows[0]),(rows[0]+1,h)]; panels += [(b-a,w) for a,b in spans if a<b]
    return bool(rows), bool(cols), len(set(panels))==1 and len(panels)>=2, panels


def classify_signature(inp: Grid, out: Grid, task: dict) -> tuple[str, list[str]]:
    ih,iw=_shape(inp); oh,ow=_shape(out); feats=[]
    in_comps=_components(inp); out_comps=_components(out)
    sep_row, sep_col, equal_panels, panels=_separator_features(inp)
    shape_rel="unknown"
    if (ih,iw)==(oh,ow): shape_rel="same_shape"
    elif oh<=ih and ow<=iw: shape_rel="cropped"
    elif oh>=ih and ow>=iw: shape_rel="expanded"
    if (oh,ow)==(iw,ih) and ih!=iw: shape_rel="transposed_dimensions"
    if panels and (oh,ow) in panels: shape_rel="panel_sized_output"
    if any(_bbox(c) and (oh,ow)==(_bbox(c)[2]-_bbox(c)[0]+1,_bbox(c)[3]-_bbox(c)[1]+1) for c in in_comps): shape_rel="component_bbox_sized_output"
    test_shapes={_shape(as_grid(x["output"])) for x in task.get("train", [])}
    if len(test_shapes)==1 and (oh,ow) in test_shapes and shape_rel=="unknown": shape_rel="fixed_output_shape"
    feats.append(shape_rel)
    if shape_rel=="same_shape":
        changed=[(inp[r][c],out[r][c]) for r in range(ih) for c in range(iw) if inp[r][c]!=out[r][c]]
        if changed:
            if all(a and b and a!=b for a,b in changed): feats.append("recolor_only")
            elif all(a==0 and b!=0 for a,b in changed): feats.append("add_only")
            elif all(a!=0 and b==0 for a,b in changed): feats.append("delete_only")
            elif len(in_comps)==len(out_comps): feats.append("move_or_copy_like")
            else: feats.append("mixed")
        feats += delta_diagnostics(inp,out)[:3]
    feats.append(f"input_components={len(in_comps)}")
    feats.append(f"output_components={len(out_comps)}")
    if sorted(map(len,in_comps))==sorted(map(len,out_comps)): feats.append("components_preserved_approximately")
    if {frozenset(c) for c in in_comps} & {frozenset(c) for c in out_comps}: feats.append("component_masks_preserved")
    if sorted(map(len,in_comps))!=sorted(map(len,out_comps)): feats.append("component_sizes_changed")
    if len(out_comps)<len(in_comps): feats.append("new_connections_created")
    if sum(v==0 for row in out for v in row) < sum(v==0 for row in inp for v in row): feats.append("holes_possibly_filled")
    if sep_row: feats.append("separator_row")
    if sep_col: feats.append("separator_column")
    if equal_panels: feats.append("equal_panels")
    if panels and (oh,ow) in panels: feats.append("output_equals_one_panel_shape")
    if panels and (oh,ow)==panels[0]: feats.append("output_shape_equals_overlay_panel_shape")
    if _exact_small_period(inp): feats.append("input_has_exact_small_period")
    if _exact_small_period(out): feats.append("output_has_exact_small_period")
    if oh>=ih and ow>=iw and _exact_small_period(out): feats.append("output_may_be_periodic_extension")
    ip,op=_palette(inp),_palette(out)
    if ip==op: feats.append("same_palette")
    elif op<ip: feats.append("subset_palette")
    else:
        if op-ip: feats.append("new_color_added")
        if ip-op: feats.append("colors_removed")
    if shape_rel=="same_shape":
        mapping={}; ok=True
        for r in range(ih):
            for c in range(iw):
                a,b=inp[r][c],out[r][c]
                if a in mapping and mapping[a]!=b: ok=False
                mapping[a]=b
        if ok: feats.append("global_color_map_possible")
    return shape_rel, feats


def semantic_class_for(family: str, rule_name: str) -> str:
    text=f"{family} {rule_name}"
    if any(x in text for x in ("panel","strip","overlay")): return "panel"
    if any(x in text for x in ("recolor","replace","color_map")): return "recolor"
    if any(x in text for x in ("periodic","tile","repeat")): return "periodicity"
    if any(x in text for x in ("dilate","extend","ray","connect")): return "local_growth"
    if any(x in text for x in ("component","object","bbox","crop")): return "object"
    if any(x in text for x in ("align","translate","mirror","d4","rot","flip","pad")): return "geometry"
    return "fallback"


def rule_metadata(family: str, name: str) -> dict[str, str]:
    view="identity"
    if "gen_d4_" in name:
        view=name.split("__",1)[0].replace("gen_d4_","")
    color="color_map" if "color_map" in name else "raw"
    direct="composed" if "__" in name or "then" in name else "direct"
    return {"family":family,"rule_name":name,"semantic_class":semantic_class_for(family,name),"view_transform":view,"color_abstraction_mode":color,"direct_or_composed":direct}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline experimental ARC rule-discovery lab.")
    parser.add_argument("--all", action="store_true", help="Run all families (default).")
    parser.add_argument("--family", action="append", help="Family name or alias to run; repeatable.")
    parser.add_argument("--task", help="Run one task id only.")
    parser.add_argument("--limit", type=int, help="Limit number of tasks processed.")
    parser.add_argument("--only-new-hits", action="store_true", help="Hide candidates with zero new hits.")
    parser.add_argument("--challenges", default="data/arc-agi_training_challenges.json")
    parser.add_argument("--solutions", default="data/arc-agi_training_solutions.json")
    parser.add_argument("--diagnostics", action="store_true", help="Report leave-one-out, support grouping, and compact residuals for evidenced candidates.")
    parser.add_argument("--signatures", action="store_true", help="Report compact signatures for production-unsolved outputs and exit.")
    parser.add_argument("--counterfactual", action="store_true", help="Estimate broad precedence risks for candidate new hits.")
    parser.add_argument("--negative-controls", action="store_true", help="Run lightweight negative-control exact-fit checks for selected family.")
    parser.add_argument("--inspect-residual", metavar="TASK_ID", help="Print detailed residual/train-pair inspection for a task and exit.")
    parser.add_argument("--show-grids", action="store_true", help="With --inspect-residual, include full train grids.")
    args = parser.parse_args()

    families = select_families(None if args.all or not args.family else args.family)
    challenges = load_json(ROOT / args.challenges)
    solutions = load_json(ROOT / args.solutions)

    if args.inspect_residual:
        inspect_residual_task(args.inspect_residual, challenges, show_grids=args.show_grids)
        return 0

    items = list(challenges.items())
    if args.task:
        items = [(tid, task) for tid, task in items if tid == args.task]
        if not items:
            raise SystemExit(f"Task not found: {args.task}")
    if args.limit is not None:
        items = items[: args.limit]

    stats: dict[tuple[str, str], CandidateStats] = {}
    production_summary = ScoreSummary()
    candidate_summary = ScoreSummary()
    candidate_new_summary = ScoreSummary()
    production_hits = production_total = 0
    all_errors = 0
    invalid_reasons_total: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    signature_examples: dict[str, list[str]] = defaultdict(list)
    counterfactual = Counter()

    for task_id, task in items:
        if task_id not in solutions:
            continue
        expected = solutions[task_id]
        prod_attempts, _ = predict_task(task)
        ph, pt = score_task_attempts(prod_attempts, expected)
        production_hits += ph
        production_total += pt
        production_summary.add_task(ph, pt)
        prod_hit_idxs = hit_indices(prod_attempts, expected)
        if args.signatures:
            for ti, item in enumerate(task["test"]):
                if ti in prod_hit_idxs or ti >= len(expected):
                    continue
                inp = as_grid(item["input"])
                out = as_grid(expected[ti])
                _, feats = classify_signature(inp, out, task)
                key = " + ".join(feats[:8])
                signature_counts[key] += 1
                ex = signature_examples[key]
                if len(ex) < 5:
                    ex.append(f"{task_id}#{ti}")

        if args.signatures:
            continue

        train = train_pairs(task)
        task_support: dict[tuple[int, tuple[tuple[int, ...], ...]], list[dict[str, str]]] = {}
        task_predictions: dict[tuple[str, str], tuple[list[Grid | None], set[int], set[int]]] = {}
        for family in families:
            try:
                rules = FAMILY_FITTERS[family](train)
            except Exception:
                all_errors += 1
                continue
            for rule in rules:
                key = (family, rule.name)
                rec = stats.setdefault(key, CandidateStats(family=family, name=rule.name))
                attempts, errors, invalid_reasons = candidate_attempts(rule, task)
                rec.errors += errors
                rec.invalid_reasons.update(invalid_reasons)
                all_errors += errors
                invalid_reasons_total.update(invalid_reasons)
                ch, _ = score_task_attempts(attempts, expected)
                cand_hit_idxs = hit_indices(attempts, expected)
                new_idxs = cand_hit_idxs - prod_hit_idxs
                overlap_idxs = cand_hit_idxs & prod_hit_idxs
                rec.hits += ch
                rec.new += len(new_idxs)
                rec.overlap += len(overlap_idxs)
                candidate_summary.add_task(ch, len(expected))
                candidate_new_summary.add_task(len(new_idxs), len(expected))
                if args.counterfactual and new_idxs:
                    counterfactual["new_hits"] += len(new_idxs)
                    counterfactual["overlap_hits"] += len(overlap_idxs)
                    counterfactual["changed_existing_solved_outputs"] += sum(1 for i in prod_hit_idxs if i < len(attempts) and not attempt_matches(attempts[i], expected[i]))
                preds_for_diag: list[Grid | None] = []
                for item in task["test"]:
                    try:
                        p = rule.predict(as_grid(item["input"]))
                    except Exception:
                        p = None
                    preds_for_diag.append(_strict_grid_or_none(p))
                if args.diagnostics:
                    for ti, pred in enumerate(preds_for_diag):
                        if pred is not None:
                            task_support[(ti, _grid_sig(pred))] = task_support.get((ti, _grid_sig(pred)), []) + [rule_metadata(family, rule.name)]
                    task_predictions[(family, rule.name)] = (preds_for_diag, cand_hit_idxs, new_idxs)
                if cand_hit_idxs:
                    rec.tasks.add(task_id)

        if args.diagnostics:
            for (family, name), (preds, cand_hit_idxs, new_idxs) in task_predictions.items():
                if not (cand_hit_idxs or new_idxs):
                    continue
                rec = stats[(family, name)]
                if new_idxs:
                    rec.loo[task_id] = _loo_status(family, name, train)
                for ti, pred in enumerate(preds):
                    if pred is None:
                        continue
                    supporters = task_support.get((ti, _grid_sig(pred)), [])
                    if len(supporters) > 1 and (ti in cand_hit_idxs or ti in new_idxs):
                        fams={m["family"] for m in supporters}; sem={m["semantic_class"] for m in supporters}; views={m["view_transform"] for m in supporters}
                        weight=len(fams)+len(sem)+min(2,len(views))
                        names=sorted({m["rule_name"] for m in supporters})
                        rec.support[f"{task_id}#{ti}"] = (weight, [f"raw_support_count={len(supporters)} distinct_base_families={len(fams)} distinct_semantic_classes={len(sem)} distinct_view_transforms={len(views)} independence_weighted_support={weight}"] + names)
                    if ti < len(expected) and ti not in cand_hit_idxs:
                        rec.residuals[f"{task_id}#{ti}"] = _residual_summary(pred, expected[ti])

    if args.signatures:
        print_score_summary("Production baseline", production_summary)
        print()
        print("Unsolved signature clusters:")
        for key, count in signature_counts.most_common(25):
            print(f"  {count} outputs: {key} examples={', '.join(signature_examples[key])}")
        print()
        print("Delta diagnostics are included in same-shape cluster labels when present.")
        return 0

    if args.negative_controls:
        print("Negative controls: deferred (TODO: add shuffle/permutation/perturbation harness without changing candidate fitting).")

    rows = sorted(stats.values(), key=lambda r: (-r.new, -r.hits, r.family, r.name))
    if args.only_new_hits:
        rows = [r for r in rows if r.new > 0]

    print("Family | Rule Name | Hits | New | Overlap | Tasks")
    print("--- | --- | ---: | ---: | ---: | ---")
    for r in rows:
        task_list = ", ".join(sorted(r.tasks)) if r.tasks else "-"
        err = f" errors={r.errors}" if r.errors else ""
        print(f"{r.family} | {r.name} | {r.hits} | {r.new} | {r.overlap} | {task_list}{err}")
        if args.diagnostics:
            if r.loo:
                print(f"  diagnostics: LOO pass: {'; '.join(f'{k}={v}' for k, v in sorted(r.loo.items()))}")
            for key, (weight, names) in sorted(r.support.items()):
                meta = names[0] if names else f"independence_weighted_support={weight}"
                rule_names = names[1:] if names else []
                print(f"  diagnostics: {key} {meta} supporting_rule_names={', '.join(rule_names[:12])}{' ...' if len(rule_names) > 12 else ''}")
            for key, summary in sorted(r.residuals.items()):
                print(f"  diagnostics: {key} residual {summary}")

    candidate_total = sum(r.hits for r in stats.values())
    candidate_new = sum(r.new for r in stats.values())
    print()
    print(f"Production baseline hits: {production_hits}/{production_total}")
    print_score_summary("Production baseline", production_summary)
    print_score_summary("Candidate totals", candidate_summary)
    print_score_summary("Candidate new hits", candidate_new_summary)
    print(f"Candidate total hits: {candidate_total}")
    print(f"Candidate new hits: {candidate_new}")
    print(f"Errors: {all_errors}")
    if invalid_reasons_total:
        print("Invalid grids: " + ", ".join(f"{k}={v}" for k, v in sorted(invalid_reasons_total.items())))
    if args.counterfactual:
        lost = counterfactual.get("changed_existing_solved_outputs", 0)
        zone = "late" if lost else "early/middle/late"
        print("Counterfactual precedence diagnostic:")
        print(f"  new_hits={counterfactual.get('new_hits', 0)}")
        print(f"  possible_lost_hits={lost}")
        print(f"  overlap_hits={counterfactual.get('overlap_hits', 0)}")
        print(f"  changed_existing_solved_outputs={lost}")
        print(f"  recommended_safe_priority_zone={zone}")
    print("Best candidates to promote:")
    promoted = [r for r in sorted(stats.values(), key=lambda r: (-r.new, -r.hits, r.name)) if r.new > 0]
    if promoted:
        for r in promoted[:20]:
            print(f"- {r.family} / {r.name}: new={r.new}, hits={r.hits}, tasks={', '.join(sorted(r.tasks))}")
    else:
        print("- None found in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
