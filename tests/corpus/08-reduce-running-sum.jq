reduce .[] as $row ([];
  . + [{
    label: $row.label,
    delta: $row.value,
    running: ((.[-1].running // 0) + $row.value)
  }]
)
