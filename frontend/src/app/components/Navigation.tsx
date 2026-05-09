"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/test", label: "Test" },
  { href: "/about", label: "About Us" },
  { href: "/report", label: "Reports" }
]

export default function Navigation() {
  const pathname = usePathname()

  return (
    <nav>
      {navLinks.map(link => (
        <Link
          key={link.href}
          href={link.href}
          className={pathname === link.href ? "active" : ""}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  )
}
