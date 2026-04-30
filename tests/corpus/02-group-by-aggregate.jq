.orders
| group_by(.customer)
| map({
    customer: .[0].customer,
    total: map(.amount) | add,
    count: length
  })
| sort_by(.total)
| reverse
