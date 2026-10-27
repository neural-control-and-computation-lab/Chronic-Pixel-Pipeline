import argparse,json,runpy,sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    catalog=json.loads((root/'workflows/catalog.json').read_text(encoding='utf8'))
    if len(sys.argv)==1 or sys.argv[1]=='--list':
        for group,entries in catalog.items():
            print(group+':')
            for name,path in entries.items():print(f'  {name}: {path}')
        return
    name=sys.argv[1]; entries={k:v for group in catalog.values() for k,v in group.items()}
    if name not in entries:raise SystemExit(f'Unknown workflow: {name}. Use --list.')
    target=root/entries[name]
    sys.path.insert(0,str(root));sys.argv=[str(target),*sys.argv[2:]]
    runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':main()
