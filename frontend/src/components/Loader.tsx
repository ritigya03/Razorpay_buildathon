export function Loader({ progress, done }: { progress: number; done: boolean }) {
  return (
    <div className={"loader" + (done ? " gone" : "")}>
      <div className="core">
        <div className="mark">S</div>
        <div className="name">Project Sentinel</div>
        <div className="track"><i style={{ width: `${Math.min(progress, 100)}%` }} /></div>
      </div>
    </div>
  );
}
