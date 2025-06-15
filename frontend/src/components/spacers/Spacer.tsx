type Props = {
  px: number;
  horizontal?: boolean;
};

export const Spacer = ({ px, horizontal }: Props) => {
  const width = !horizontal ? 1 : px;
  const height = horizontal ? 1 : px;

  return (
    <span
      style={{
        display: "block",
        width,
        minWidth: width,
        height,
        minHeight: height,
      }}
    />
  );
};
