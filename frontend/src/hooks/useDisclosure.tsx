import { useCallback, useState } from "react";

type Props = {
  initialState?: boolean;
  onOpen?: () => void;
  onClose?: () => void;
};

export const useDisclosure = ({ initialState = false, onOpen, onClose }: Props) => {
  const [isOpen, setIsOpen] = useState<boolean>(initialState);

  const open = useCallback(() => {
    onOpen && onOpen();
    setIsOpen(true);
  }, [onOpen]);

  const close = useCallback(() => {
    onClose && onClose();
    setIsOpen(false);
  }, [onClose]);

  const toggle = useCallback(() => {
    if (isOpen) {
      onClose && onClose();
      setIsOpen(false);
    } else {
      onOpen && onOpen();
      setIsOpen(true);
    }
  }, [isOpen, onOpen, onClose]);

  return {
    isOpen,
    open,
    close,
    toggle,
  };
};
