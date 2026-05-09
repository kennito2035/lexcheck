import ReportClient from "./ReportClient"

export default async function ReportPage({
  searchParams,
}: {
  searchParams: Promise<{ rid?: string }>
}) {
  const { rid } = await searchParams
  return (
    <>
      <h1>Reports</h1>
      <ReportClient key={rid ?? "__none__"} rid={rid ?? null} />
      <p>Disclaimer: this tool provides risk indicators and is not a diagnosis.</p>
    </>
  )
}

