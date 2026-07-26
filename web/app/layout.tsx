import "./globals.css";

export const metadata = { title: "rho — résumé editor" };

// Set the theme before first paint so there's no flash of the wrong palette.
// Reads the saved choice, falls back to the OS preference.
const themeInit = `(function(){try{var t=localStorage.getItem("rho-theme");if(!t)t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.setAttribute("data-theme",t);}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeInit }} /></head>
      <body>{children}</body>
    </html>
  );
}
