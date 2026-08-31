/**
 * FilmOry logo icon — renders the brand logo from /logo.jpeg.
 */
export function FilmoryLogo({
  size = 28,
  className = "",
}: {
  size?: number;
  className?: string;
}) {
  return (
    <img
      src="/logo.png"
      alt="FilmOry"
      width={size}
      height={size}
      className={className}
      aria-hidden="true"
      style={{ objectFit: "contain" }}
    />
  );
}
