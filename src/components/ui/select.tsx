import { Portal } from "@ark-ui/react/portal";
import { Select as ArkSelect, createListCollection } from "@ark-ui/react/select";
import { ChevronDownIcon } from "lucide-react";

export interface SelectOption {
  label: string;
  value: string;
}

interface SelectProps {
  label: string;
  placeholder: string;
  options: SelectOption[];
  value: string | null;
  onChange: (value: string) => void;
  hideLabel?: boolean;
}

export function Select({ label, placeholder, options, value, onChange, hideLabel }: SelectProps) {
  const collection = createListCollection({
    items: options,
    itemToValue: (item) => item.value,
    itemToString: (item) => item.label,
  });

  return (
    <ArkSelect.Root
      collection={collection}
      value={value ? [value] : []}
      onValueChange={(details) => onChange(details.value[0])}
    >
      <ArkSelect.Label className={hideLabel ? "sr-only" : "mb-2 block text-[14px] font-medium text-onyx"}>
        {label}
      </ArkSelect.Label>
      <ArkSelect.Control>
        <ArkSelect.Trigger className="flex h-11 w-full items-center justify-between rounded-lg border border-faint-line bg-pure-white px-4 text-[16px] text-onyx outline-none transition-colors focus:border-signal-blue focus:ring-2 focus:ring-signal-blue/25 data-[state=open]:border-signal-blue">
          <ArkSelect.ValueText placeholder={placeholder} className="whitespace-nowrap text-left text-onyx" />
          <ArkSelect.Indicator>
            <ChevronDownIcon className="h-4 w-4 text-warm-gray" />
          </ArkSelect.Indicator>
        </ArkSelect.Trigger>
      </ArkSelect.Control>
      <Portal>
        <ArkSelect.Positioner>
          <ArkSelect.Content className="z-50 min-w-(--reference-width) overflow-hidden rounded-lg border border-faint-line bg-pure-white shadow-subtle">
            {collection.items.map((item) => (
              <ArkSelect.Item
                key={item.value}
                item={item}
                className="flex cursor-pointer select-none items-center justify-between px-4 py-3 text-[16px] text-onyx data-highlighted:bg-paper-white data-[state=checked]:font-medium"
              >
                <ArkSelect.ItemText>{item.label}</ArkSelect.ItemText>
                <ArkSelect.ItemIndicator className="text-signal-blue">✓</ArkSelect.ItemIndicator>
              </ArkSelect.Item>
            ))}
          </ArkSelect.Content>
        </ArkSelect.Positioner>
      </Portal>
      <ArkSelect.HiddenSelect />
    </ArkSelect.Root>
  );
}
