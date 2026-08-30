export function Loader({ progress, done }: { progress: number; done: boolean }) {
  return (
    <div className={"loader" + (done ? " gone" : "")}>
      <div className="core">
        <div className="rings">
          <span /><span /><span />
          <i className="bead" />
        </div>
        <div className="word">Sentinel</div>
        <div className="sub">privacy-preserving fraud-ring intelligence</div>
        <div className="track"><i style={{ width: `${Math.min(progress, 100)}%` }} /></div>
      </div>
    </div>
  );
}
