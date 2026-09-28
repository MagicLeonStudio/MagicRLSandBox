import { Link, NavLink } from "react-router";

const navItems = [
  { to: "/", label: "首页" },
  { to: "/tutorials", label: "教程" },
  { to: "/playground", label: "训练场" },
];

export default function Navbar() {
  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 h-14 flex items-center justify-between gap-4">
        <Link to="/" className="flex items-center gap-2">
          <span className="font-bold text-xl text-accent-yellow">MagicRL SandBox</span>
          <span className="text-muted-foreground text-xs font-mono hidden sm:inline">v0.8</span>
        </Link>
        <div className="flex items-center gap-1">
          {navItems.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm transition-colors ${
                  isActive
                    ? "text-accent-yellow bg-accent-purple/15"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent-purple/10"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <a
            href="https://github.com/MagicLeonStudio/MagicRLSandBox"
            target="_blank"
            rel="noreferrer"
            className="ml-2 text-xs text-muted-foreground hover:text-accent-purple font-mono"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}
