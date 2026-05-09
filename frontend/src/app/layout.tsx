import type { Metadata } from "next"
import Link from "next/link"
import type { ReactNode } from "react"

import { Inter, Titan_One } from "next/font/google"

import "./globals.css"
import Navigation from "./components/Navigation"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })
const titanOne = Titan_One({ 
  weight: "400", 
  subsets: ["latin"],
  variable: "--font-bubble"
})

export const metadata: Metadata = {
  title: "LexCheck",
  description: "LexCheck"
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${titanOne.variable}`} suppressHydrationWarning>
        <header>
          <div>
            <Link href="/">LexCheck</Link>
            <Navigation />
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  )
}
